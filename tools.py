import csv
import io
import pickle

# The only objects a SOM model file saved by this app can contain
_SOM_PICKLE_ALLOWED = {
    ('minisom', 'MiniSom'), ('builtins', 'getattr'),
    ('numpy', 'ndarray'), ('numpy', 'dtype'),
    ('numpy._core.multiarray', '_reconstruct'), ('numpy.core.multiarray', '_reconstruct'),
    ('numpy.random._pickle', '__randomstate_ctor'), ('numpy.random._pickle', '__bit_generator_ctor'),
    ('numpy.random._mt19937', 'MT19937'),
}


class _SOMUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in _SOM_PICKLE_ALLOWED:
            raise pickle.UnpicklingError(
                f"'{module}.{name}' is not allowed in a SOM model file")
        return super().find_class(module, name)


def load_som_model(file):
    """Load a [som, features] file downloaded from this app, refusing anything else."""
    return _SOMUnpickler(file).load()


def dict_to_csv(data_dict):
    output = io.StringIO()

    # Write each key and its list of values in a separate line
    for key, values in data_dict.items():
        # convert list of values in integer
        key = [int(k) for k in key]
        values = [int(value) for value in values]
        row = f"{key}: {values}\n"
        output.write(row)

    return output.getvalue()
