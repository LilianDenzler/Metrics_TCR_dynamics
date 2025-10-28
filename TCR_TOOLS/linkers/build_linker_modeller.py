from modeller import Environ
from modeller.automodel import LoopModel, refine
from modeller import selection
from modeller import environ
from modeller import log



import argparse

class ScFvModel(LoopModel):
    def __init__(self, *args, linker_pos_start=None, linker_pos_end=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.linker_pos_start = int(linker_pos_start)
        self.linker_pos_end = int(linker_pos_end)

    def select_loop_atoms(self):
        # Only remodel the linker region
        return selection(self.residue_range(f'{self.linker_pos_start}:A', f'{self.linker_pos_end}:A'))

import os

def build_model( linker_dir, alignment_file, model_name,
                linker_pos_start, linker_pos_end):
    env = Environ()
    env.io.atom_files_directory = [linker_dir]
    a = ScFvModel(env,
                    alnfile=alignment_file,
                    knowns=model_name,
                    sequence=model_name + "_output",
                    linker_pos_start=linker_pos_start,
                    linker_pos_end=linker_pos_end)

    a.starting_model = 1
    a.ending_model = 1
    a.loop.starting_model = 1
    a.loop.ending_model = 1
    a.loop.md_level = refine.slow

    a.make()


if __name__=="__main__":
    #pass arguments
    MODELLER_KEY="MODELIRANJE"

    parser = argparse.ArgumentParser(description='Modeller for scFv')
    parser.add_argument('linker_dir', type=str, help='Directory containing linker')
    parser.add_argument('model_name', type=str, help='Model name')
    parser.add_argument('alignment_file', type=str, help='Alignment file')
    parser.add_argument('linked_pos_start', type=int, help='start of linker position')
    parser.add_argument('linked_pos_end', type=int, help='end of linker position')
    parser.add_argument('output_dir', type=str, help='Output directory')
    args = parser.parse_args()
    print(args.linked_pos_start, args.linked_pos_end)
    original_cwd = os.getcwd()
    os.chdir(args.output_dir)
    build_model( args.linker_dir, args.alignment_file, args.model_name, args.linked_pos_start, args.linked_pos_end)
    os.chdir(original_cwd)