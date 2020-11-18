import argparse
import shutil
import subprocess
from pathlib import Path
import json
import re

train_dir_name = 'train'
validation_dir_name = 'validation'

model_final_name = 'model.tar'
classes_file = 'classes.txt'
class_map_file = 'class_map.json'

singular_type = 'singular'
model_type = 'model'

models = {
    25: 'tf_mobilenetv3_small_100',
    50: 'mobilenetv2_140',
    100: 'resnet26'
}


def get_classes(check_path: str) -> [str]:
    file = open(check_path, 'r')
    class_names = file.readlines()
    class_names.sort()
    class_names = map(lambda l: str.replace(l, '\n', ''), class_names)
    return list(class_names)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str)
    parser.add_argument("--target", type=str)
    parser.add_argument("--classes-path", type=str, default=None)
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--num-gpus", type=int, default=1)
    args = parser.parse_args()
    return args


def run_command(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    full_out = ''
    while True:
        output = process.stdout.readline().decode("utf-8")
        full_out += output
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
    return full_out


def create_model_map():
    argv = parse_args()

    source = Path(argv.source)
    target = Path(argv.target)

    result_path = target.joinpath('class_map.json')
    model_artefacts_path = target.joinpath('artefacts')

    class_map = {}
    if result_path.exists():
        with result_path.open('r') as json_file:
            class_map = json.load(json_file)

    classes = get_classes(argv.classes_path)

    for clazz in classes:
        if clazz in class_map:
            continue

        class_root = source.joinpath(clazz)
        train_dir = class_root.joinpath(train_dir_name)

        sub_classes = [x.name for x in train_dir.iterdir() if x.is_dir()]
        sub_classes.sort()
        class_size = len(sub_classes)

        if class_size == 1:
            class_map[clazz] = {
                "type": singular_type,
                "value": sub_classes[0],
                "confidence": 1
            }
            continue
        else:
            if class_size <= 25:
                model_arch = models[25]
            elif class_size <= 50:
                model_arch = models[50]
            else:
                model_arch = models[100]

        training_process_output = run_command([
            "./distributed_train.sh",
            str(argv.num_gpus), class_root,
            "--model", model_arch,
            "--img-size", str(argv.img_size),
            "--num-classes", str(class_size),
            "--output", model_artefacts_path,
            "--j", str(argv.num_gpus),
            "--sched", "tanh",
            "--epochs", "25",
            "--warmup-epochs", "5",
            "--lr", "0.003125",
            "--reprob", "0.5",
            "--remode", "pixel",
            "--batch-size", "16",
            "--pretrained",
            "--color-jitter", "0.7",
            "--aa", "v0",
            "--use-multi-epochs-loader",
        ])

        output = training_process_output
        interesting_output = re.search("\\*\\*\\* Best metric: <.+?>, epoch: <.+?>, path: <.+?> \\*\\*\\*", output)
        if interesting_output is None:
            print(f"Bad output for {clazz}, terminating")
            exit(-1)

        results = interesting_output.group()
        precision = float(re.search("(?<=Best metric: <).+?(?=>)", results).group())
        path = re.search("(?<=path: <).+?(?=>)", results).group()

        output_model_artifact_path = Path(path).joinpath('model_best.pth.tar')

        target_path = target.joinpath(clazz)
        model_final_path = target_path.joinpath(model_final_name)
        classes_txt_path = target_path.joinpath(classes_file)
        class_map_json_path = target_path.joinpath(class_map_file)

        shutil.copy(output_model_artifact_path, model_final_path)

        with classes_txt_path.open("w") as f:
            f.writelines("\n".join(sub_classes))

        sub_class_map = {sub_classes[i]: sub_classes[i].replace("_", " ") for i in range(0, class_size)}
        with class_map_json_path.open("w") as f:
            json.dump(sub_class_map, f)

        class_map[clazz] = {
            "type": model_type,
            "value": target_path,
            "confidence": round(precision, 2)
        }

    with result_path.open("w") as f:
        json.dump(class_map, f)

    return


if __name__ == "__main__":
    create_model_map()
