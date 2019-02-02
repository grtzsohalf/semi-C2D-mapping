import torch
import os


class PytorchSaver():

    def __init__(self, keep_num, save_dir):
        assert(keep_num >= 1)
        self.keep_num = int(keep_num)
        self.save_dir = save_dir
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
        self.checkpoint = []  # [old(low epoch).......new(high epoch)]

    def save(self, state, file_name):
        torch.save(state, os.path.join(self.save_dir, file_name))
        self._update_checkpoint(file_name)

    @staticmethod
    def load_dir(save_dir):
        with open(os.path.join(save_dir, 'checkpoint'), 'r') as f:
            model_name = f.read().splitlines()[0]
        return model_name, torch.load(os.path.join(save_dir, model_name))

    @staticmethod
    def load_path(path):
        return torch.load(path)

    def _update_checkpoint(self, name):
        if len(self.checkpoint) == self.keep_num:
            os.remove(os.path.join(self.save_dir, self.checkpoint[0]))
            for i in range(self.keep_num-1):
                self.checkpoint[i] = self.checkpoint[i+1]
            self.checkpoint[self.keep_num-1] = name
        else:
            self.checkpoint.append(name)

        checkpoint = ''
        for history in reversed(self.checkpoint):
            checkpoint += history
            checkpoint += '\n'
        with open(os.path.join(self.save_dir, 'checkpoint'), 'w') as f:
            f.write(checkpoint)


if __name__ == '__main__':
    saver = PytorchSaver(10, 'model')
    for i in range(100):
        state = {'index':i}
        saver.save(state, str(i))
    print(PytorchSaver.load_dir('model')['index'])

