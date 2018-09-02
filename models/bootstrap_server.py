"""
This server will do the following:
1. Check environment variable for batching parameters 
2. Write a batching_parameter_file
3. Run tensorflow_serving executable with batching enabled and pass in all argv
"""

from subprocess import call
import sys
import os
from shlex import split

def _get_batching_params(max_batch_size,
                         batch_timeout_micros=10000,
                         max_enqueued_batches=200):
    batching_params_text = ("max_batch_size {{ value : {mbs} }} \n"
                            "batch_timeout_micros {{ value : {btm} }} \n"
                            "max_enqueued_batches {{ value : {meb} }} \n"
                            "num_batch_threads {{ value : {nbt} }} \n")

    num_batch_threads = max_batch_size * 2

    formatted_params = batching_params_text.format(
        mbs=max_batch_size,
        btm=batch_timeout_micros,
        meb=max_enqueued_batches,
        nbt=num_batch_threads)

    return formatted_params


def main():
    max_batch_size = os.environ.get('max_batch_size', 1)
    batch_timeout_micros = os.environ.get('batch_timeout_micros', 10000)
    max_enqueued_batches = os.environ.get('max_enqueued_batches', 200)

    with open('bpf.json', 'w') as f:
        text = _get_batching_params(max_batch_size, batch_timeout_micros,
                                    max_enqueued_batches)
        print("Batching Params: \n", text)
        f.write(text)

    cmd = [
            "tensorflow_model_server", "--enable_batching",
            "--batching_parameters_file=bpf.json"
        ] + sys.argv[1:]
    cmd = ' '.join(cmd)
    print(cmd)
    call(cmd,shell=True)


if __name__ == '__main__':
    main()
