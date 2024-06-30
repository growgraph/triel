import pathlib
from pprint import pprint

import click
import requests
import suthing


@click.command()
@click.option("--port", type=click.INT, default=8888)
@click.option("--host", type=click.STRING, default="localhost")
@click.option("--service", type=click.STRING, default="bern")
@click.option("--input-path", type=click.Path(path_type=pathlib.Path))
@click.option("--output", type=click.Path(path_type=pathlib.Path), required=False)
def run(host, port, service, input_path, output):
    inp = suthing.FileHandle.load(fpath=input_path)
    if service == "bern":
        route = "plain"
    elif service == "fishing":
        route = "service/disambiguate"
    elif service == "pelinker":
        route = "pelinker"
    else:
        raise ValueError(f"unsupported service {service}")

    url = f"http://{host}:{port}/{route}"
    r = requests.post(url, json={"text": inp["text"]}, verify=False).json()
    pprint(r)
    if output:
        suthing.FileHandle.dump(r, output.as_posix())


if __name__ == "__main__":
    run()
