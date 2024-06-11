from process_data import data_prcoess
import network_building
import test_output

## this is to generate the training dat a and test data
## all of the data has been stored in the folder"Inputs to network"
train_loader,test_loader,val_loader = data_prcoess(img_path='D:\Apps\CNN_project-2\data',batch_size=1)
##
## start training
## after the training, the trained network will be saved, the file name is returned
## the record loss value per epoch will also be saved
trained_network_name=network_building.build_network(train_loader=train_loader,val_loader=val_loader,epochs=20)

## this is for test
## input the test_inputs, the return will be the output of the trained network.
predict_res =test_output.trained_network_test(test_loader=test_loader,trained_network_name=trained_network_name)
print('------test results-------\n')
print(f'正确率：{(predict_res[0][0]+predict_res[1][1]+predict_res[2][2])/sum([num for row in predict_res for num in row])}')
with open('predict_res_2.txt', 'w') as file:
    file.write(str(predict_res))