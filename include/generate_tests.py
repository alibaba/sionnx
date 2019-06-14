import os
import glob

os.system("cp ../build/tblgen/bin/llvm-tblgen .")
dir_path = os.path.dirname(os.path.realpath(__file__))
td_files = glob.glob(os.path.join(dir_path, '*.td'))
lens = len(td_files)
for k in range(lens):
    base = os.path.basename(td_files[k])
    out_file_name = os.path.splitext(base)[0]
    os.system("./llvm-tblgen -gen-onnx-tests " + td_files[k] + " -I ./ -o ./tests/" + out_file_name + ".py") 
    print(out_file_name + ".py generated.")
