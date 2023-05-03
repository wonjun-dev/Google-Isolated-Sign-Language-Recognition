import torch


class Graph:
    def __init__(self, hop=1) -> None:
        self.g_lhand = {
            0: [1, 5, 17],
            1: [0, 2],
            2: [1, 3],
            3: [2, 4],
            4: [3],
            5: [0, 6, 9],
            6: [5, 7],
            7: [6, 8],
            8: [7],
            9: [5, 10, 13],
            10: [9, 11],
            11: [10, 12],
            12: [11],
            13: [9, 14, 17],
            14: [13, 15],
            15: [14, 16],
            16: [15],
            17: [0, 13, 18],
            18: [17, 19],
            19: [18, 20],
            20: [19],
        }
        self.g_rhand = self.g_lhand
        self.g_spose = {
            0: [1, 2],
            1: [0],
            2: [0],
            3: [4, 5, 9],
            4: [3, 6, 10],
            5: [3, 7],
            6: [4, 8],
            7: [5],
            8: [6],
            9: [3, 10],
            10: [9, 4],
        }
        num_joints = (
            len(self.g_lhand.keys())
            + len(self.g_rhand.keys())
            + len(self.g_spose.keys())
        )
        self.A = torch.zeros((num_joints, num_joints), requires_grad=False)
        part_offset = [0, 21, 42]
        for part, part_offset in zip(
            [self.g_lhand, self.g_rhand, self.g_spose], part_offset
        ):
            for k, neighbor in part.items():
                self.A[k + part_offset][torch.tensor(neighbor) + part_offset] = 1

        if hop == 2:
            for i in range(num_joints):
                idxs = torch.nonzero(self.A[i] == 1).squeeze().tolist()
                if isinstance(idxs, int):
                    idxs = [idxs]
                for j in idxs:
                    connection = torch.nonzero(self.A[j] == 1).squeeze()
                    self.A[i][connection] = 1

        # self coneection
        self.A = self.A + torch.eye(num_joints)


# class GraphB:
#     def __init__(self) -> None:
#         self.g_lhand = {
#             0: [1, 5, 17, 4, 8, 12, 16, 20],
#             1: [0, 2, 5, 9, 13, 17],
#             2: [1, 3, 6, 10, 14, 18],
#             3: [2, 4, 7, 11, 15, 19],
#             4: [3, 8, 12, 16, 20, 0],
#             5: [0, 6, 9, 1, 13, 17],
#             6: [5, 7, 2, 10, 14, 18],
#             7: [6, 8, 3, 11, 15, 19],
#             8: [7, 4, 12, 16, 20, 0],
#             9: [5, 10, 13, 1, 17],
#             10: [9, 11, 2, 6, 14, 18],
#             11: [10, 12, 3, 7, 15, 19],
#             12: [11, 4, 8, 16, 20, 0],
#             13: [9, 14, 17, 1, 5],
#             14: [13, 15, 2, 6, 10, 18],
#             15: [14, 16, 3, 7, 11, 19],
#             16: [15, 4, 8, 12, 20, 0],
#             17: [0, 13, 18, 1, 5, 9],
#             18: [17, 19, 2, 6, 10, 14],
#             19: [18, 20, 3, 7, 11, 15],
#             20: [19, 4, 8, 12, 16, 0],
#         }
#         self.g_rhand = self.g_lhand
#         self.g_spose = {
#             0: [1, 2, 7, 8],
#             1: [0],
#             2: [0],
#             3: [4, 5, 9, 6, 7, 8],
#             4: [3, 6, 10, 5, 7, 8],
#             5: [3, 7, 4, 6, 8, 9, 10],
#             6: [4, 8, 3, 5, 7, 9, 10],
#             7: [5, 3, 4, 6, 8, 9, 10],
#             8: [6, 3, 4, 5, 7, 9, 10],
#             9: [3, 10, 5, 6, 7, 8],
#             10: [9, 4, 5, 6, 7, 8],
#         }
#         num_joints = (
#             len(self.g_lhand.keys())
#             + len(self.g_rhand.keys())
#             + len(self.g_spose.keys())
#         )
#         self.A = torch.zeros((num_joints, num_joints), requires_grad=False)
#         part_offset = [0, 21, 42]
#         for part, offset in zip(
#             [self.g_lhand, self.g_rhand, self.g_spose], part_offset
#         ):
#             for k, neighbor in part.items():
#                 self.A[k + offset][torch.tensor(neighbor) + offset] = 1

#         # self coneection
#         self.A = self.A + torch.eye(num_joints)
#         # body (wrist) and hand connection
#         self.A[0][7 + part_offset[-1]], self.A[7 + part_offset[-1]][0] = 1, 1  # lhand
#         self.A[21][8 + part_offset[-1]], self.A[8 + part_offset[-1]][21] = 1, 1  # rhand


if __name__ == "__main__":
    g = Graph(hop=2)
    # print(g.A)
    # print(g.A[0])
    # print(g.A[:, 0])
