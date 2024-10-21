from resdata import ResDataType

grdecl_file_base_path = "../../Downloads/"
# short_file_name_in = 'TARGET_MODEL_REAL2_singlefile.grdecl'
# short_file_name_out = 'TARGET_MODEL_REAL2_singlefile_corrected.grdecl'

short_file_name_in = '50x50x50_Box_Singlefile.grdecl'
short_file_name_out = '50x50x50_Box_Singlefile_corr.grdecl'


# Paths to your input and output files
input_file_path = grdecl_file_base_path + short_file_name_in
corrected_output_file_path = grdecl_file_base_path + short_file_name_out

# Define your mapping of words to be replaced
mapping_to_short = {
    'Job1_PORO': {
        'short': 'PORO',
        'type': ResDataType.RD_FLOAT
    },
    'ConnectedChannels_only': {
        'short': 'CONNECTC',
        'type': ResDataType.RD_INT
    },
    'Channel': {
        'short': 'CHANNEL',
        'type': ResDataType.RD_INT
    },
    'Formation_factor': {
        'short': 'FFACTOR',
        'type': ResDataType.RD_FLOAT
    },
    # Add more mappings as needed
}


if __name__ == '__main__':
    # Process the file line by line
    with open(input_file_path, 'r') as in_file:
        with open(corrected_output_file_path, 'w') as out_file:
            for line in in_file:
                # Replace all the words based on the dictionary mapping
                for original_word, replacement_dict in mapping_to_short.items():
                    line = line.replace(original_word, replacement_dict['short'])
                out_file.write(line)
