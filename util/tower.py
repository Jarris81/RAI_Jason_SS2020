
class Tower:

    def __init__(self, C, V, posxy=(0, 0, 0)):
        self.posXY = posxy
        self.blocks = []
        self.C = C
        self.V = V

    def add_block(self, block):
        # update the placement when adding a block
        block_pos = self.C.frame(block).getPosition()
        self.posXY[:2] = block_pos[:2]
        print(self.posXY)
        # add to tower
        self.blocks.append(block)

    def get_placement(self):
        placement = list(self.posXY)
        # sum up all blocks height
        placement[2] = -0.03 + placement[2] + sum([self.C.frame(block).info()["size"][-2] for block in self.blocks])
        return placement

    def get_blocks(self):
        return self.blocks
