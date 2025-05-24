#
# This file originates from
# https://github.com/locuslab/lcp-physics/tree/a85ecfe0fdc427ee016f3d1c2ddb0d0c0f98f21b/lcp_physics, licensed under the
# Apache License, version 2.0 (see LICENSE).
#
"""
Author: Filipe de Avila Belbute Peres
Based on: M. B. Cline, Rigid body simulation with contact and constraints, 2002
"""

import torch

from diffworld.diffsdfsim.lcp_physics.lcp.lcp import LCPFunction, LCPFunctionClass
from icecream import ic

class Engine:
    """Base class for stepping engine."""
    def solve_dynamics(self, world, dt):
        raise NotImplementedError


class PdipmEngine(Engine):
    """Engine that uses the primal dual interior point method LCP solver.
    """
    def __init__(self, max_iter=20):
        self.lcp_solver = LCPFunction
        
        self.cached_inverse = None
        self.max_iter = max_iter

    # @profile
    def solve_dynamics(self, world, dt):
        # ic('flag')
        t = world.t
        Je, B = world.Je()
        neq = Je.size(0) if Je.ndimension() > 0 else 0
        f = world.apply_forces(t)
        u = torch.matmul(world.M(), world.get_v()) + dt * f
        # ic(dt, dt*f, u)
        u = u.float()
        force = None
        if neq > 0:
            u = torch.cat([u, B])
        if not world.contacts:
            # No contact constraints, no complementarity conditions
            if neq > 0:
                P = torch.cat([torch.cat([world.M(), -Je.t()], dim=1),
                               torch.cat([Je, Je.new_zeros(neq, neq)],
                                         dim=1)])
            else:
                P = world.M()
            if self.cached_inverse is None:
                inv = torch.inverse(P)
                if world.static_inverse:
                    self.cached_inverse = inv
            else:
                inv = self.cached_inverse
            x = torch.matmul(inv, u)  # Kline Eq. 2.41
        else:
            # Solve Mixed LCP (Kline 2.7.2)
            Jc = world.Jc()
            v = torch.matmul(Jc, world.get_v()) * world.restitutions()
            M = world.M().unsqueeze(0)
            if neq > 0:
                b = B.unsqueeze(0)
                Je = Je.unsqueeze(0)
            else:
                b = torch.tensor([])
                Je = torch.tensor([])
            Jc = Jc.unsqueeze(0)
            u = u[:world.M().size(0)].unsqueeze(0)
            v = v.unsqueeze(0)
            E = world.E().unsqueeze(0)
            mu = world.mu().unsqueeze(0)
            Jf = world.Jf().unsqueeze(0)
            G = torch.cat([Jc, Jf,
                           Jf.new_zeros(Jf.size(0), mu.size(1), Jf.size(2))], dim=1)
            F = G.new_zeros(G.size(1), G.size(1)).unsqueeze(0)

            F[:, Jc.size(1):-E.size(2), -E.size(2):] = E
            F[:, -mu.size(1):, :mu.size(2)] = mu
            F[:, -mu.size(1):, mu.size(2):mu.size(2) + E.size(1)] = \
                -E.transpose(1, 2)

            # Calculate penetration depths and update h
            penetration_depths = []
            for contact in world.contacts:
                penetration_depths.append(contact[0][3])
            penetration_depths = torch.cat(penetration_depths).unsqueeze(0).to(v.device).float()

            C = world.configs['stabilization_coeff']/dt 
            copy = penetration_depths.clone()
            copy[penetration_depths < world.configs['contact_eps']/2.] = 0.
            penetration_depths = copy
            h = torch.cat([v - C*penetration_depths, v.new_zeros(v.size(0), Jf.size(1) + mu.size(1))], 1)
            x = - self.lcp_solver(max_iter=world.configs['solver_max_iter'], \
                verbose=world.configs['solver_verbose'],eps=world.configs['solver_eps'])(M, u, G, h, Je, b, F) #-1: no output 1:verbose
        new_v = x[:world.vec_len * len(world.bodies)].squeeze(0)
        return new_v, force

    def post_stabilization(self, world):
        v = world.get_v()
        M = world.M()
        Je = world.Je()
        Jc = None
        if world.contacts:
            Jc = world.Jc()
        ge = torch.matmul(Je, v)
        gc = None
        if Jc is not None:
            gc = torch.matmul(Jc, v) + torch.matmul(Jc, v) * -world.restitutions()

        u = torch.cat([Je.new_zeros(Je.size(1)), ge])
        if Jc is None:
            neq = Je.size(0) if Je.ndimension() > 0 else 0
            if neq > 0:
                P = torch.cat([torch.cat([M, -Je.t()], dim=1),
                               torch.cat([Je, Je.new_zeros(neq, neq)], dim=1)])
            else:
                P = M
            if self.cached_inverse is None:
                inv = torch.inverse(P)
            else:
                inv = self.cached_inverse
            x = torch.matmul(inv, u)
        else:
            v = gc
            Je = Je.unsqueeze(0)
            Jc = Jc.unsqueeze(0)
            h = u[:M.size(0)].unsqueeze(0)
            b = u[M.size(0):].unsqueeze(0)
            M = M.unsqueeze(0)
            v = v.unsqueeze(0)
            F = Jc.new_zeros(Jc.size(1), Jc.size(1)).unsqueeze(0)
            x = self.lcp_solver()(M, h, Jc, v, Je, b, F)
        dp = -x[:M.size(0)]
        return dp
