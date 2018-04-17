import os
import sys
import argparse
import paddle.fluid as fluid


def Transpile(src_dir, dst_dir, model_filename, params_filename):
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    inference_scope = fluid.core.Scope()
    with fluid.scope_guard(inference_scope):
        print "Loading inference_program from ", src_dir
        [inference_program, feed_target_names,
         fetch_targets] = fluid.io.load_inference_model(src_dir, exe, model_filename, params_filename)

        inference_transpiler_program = inference_program.clone()
        # NOTE: Applying the inference transpiler will change the inference_transpiler_program.
        t = fluid.InferenceTranspiler()
        t.transpile(inference_transpiler_program, inference_scope, place)

        #print inference_transpiler_program

        print "Saving the optimized inference_program to ", dst_dir
        # There is a bug in fluid.io.save_inference_model, so we can use the following code instead.
        if not os.path.isdir(dst_dir):
            os.makedirs(dst_dir)
        model_path = os.path.join(dst_dir, model_filename)
        with open(model_path, "wb") as f:
            f.write(inference_transpiler_program.desc.serialize_to_string())
        fluid.io.save_persistables(exe, dst_dir, inference_transpiler_program, params_filename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', help='Source directory of inference model')
    parser.add_argument('--dst_dir', help='Dstination directory of inference model')
    parser.add_argument('--model_filename', default=None, help='The name of model file')
    parser.add_argument('--params_filename', default=None, help='The name of params file')
    args = parser.parse_args()
    Transpile(args.src_dir, args.dst_dir, args.model_filename, args.params_filename)


if __name__ == '__main__':
    main()
