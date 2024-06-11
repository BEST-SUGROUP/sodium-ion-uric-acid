# sodium-ion-uric-acid
The main branch is the codes for classification of sodium-ion， while the uric-acid branch is the codes for classification of uric-acid.
Both branches followes the same organized structure:
network_building.py  :  construct and build the network
network_in_training_XX.pth :  parameters checkpoint we get during the training
process_data: data preprocessing and data loading
test_output.py:  test function for the final test.
main.py:  you can start the training and validation and test from here; 
          replace the img_path with the correct path of your own data in line 7
          comment line12 if you want the inference only and pass the name of XX.pth file to trained_network_name in line16.
