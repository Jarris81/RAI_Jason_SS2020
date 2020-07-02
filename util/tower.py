
class Tower:

    def __init__(self, C, V, posxy=(0, 0, 0)):
        self.posXY = posxy
        self.blocks = []
        self.C = C
        self.V = V

    def add_block(self, block):

        self.blocks.append(block)

        # change color to blue of block added to tower
        #self.C.frame(block)
        #self.V.setConfiguration(self.C)
        #self.V.recopyMeshes(self.C)

    def get_placement(self):
        placement = list(self.posXY)
        # sum up all blocks height
        placement[2] = -0.03 + placement[2] + sum([self.C.frame(block).info()["size"][-2] for block in self.blocks])
        return placement

    def get_blocks(self):
        return self.blocks
