
class Tower:

    def __init__(self, C, posxy=(0, 0, 0)):
        self.posXY = posxy
        self.blocks = []
        self.C = C

    def add_block(self, block):

        self.blocks.append(block)

        # change color to blue of block added to tower
        self.C.frame(block)

    def get_placement(self):
        placement = list(self.posXY)
        # sum up all blocks height
        if len(self.blocks):
            frame = self.C.frame(self.blocks[0])
            info = frame.info()
            print("height is: ", info)
        placement[2] = placement[2] + sum([self.C.frame(block).info()["size"][-2] for block in self.blocks])

        return placement

    def get_blocks(self):
        return self.blocks
