import csv
import io


def dict_to_csv(data_dict):
    output = io.StringIO()

    # Write each key and its list of values in a separate line
    for key, values in data_dict.items():
        # convert list of values in integer
        # values = [int(value) for value in values]
        row = f"{key}: {values}\n"
        output.write(row)

    return output.getvalue()
