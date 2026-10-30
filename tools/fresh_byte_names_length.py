"""Inject only the registered length score into the shared strict guard."""
from src.byte_name_length import LengthByteNameScorer,length_scale
from tools import byte_names_experiment as base
from tools import byte_names_length_experiment as variant
from tools.prepare_byte_names import prepare


def main():
    from tools import fresh_byte_names as common
    protocol=variant.check();fold=protocol["fresh_guard"]["fold"]
    scale=length_scale(prepare(fold)["full"])
    def factory(path):
        relative=path.relative_to(variant.FOLDER)
        return LengthByteNameScorer(base.FOLDER/relative,scale)
    common.ByteNameScorer=factory
    common.FOLDER,common.TAG=variant.FOLDER,variant.TAG
    common.check_lock=variant.check
    common.fingerprint=lambda:base.sha(variant.PROTOCOL)
    common.run()


if __name__=="__main__":main()
