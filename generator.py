"""
Out of the box pipeline to generator the final result.
"""

from inference.inferer import get_inference

print(get_inference(targ_file_path="./src/src-test/src-test.aac",
                    params_path="./model/train.6_full.pth",
                    fname="test",
                    output_folder="./inference/output"))