from matplotlib import pyplot as plt
import torch

from SupportScripts.adjustedLoss import adjustedLoss, total_loss_fuction
from SupportScripts.device import device_selecter






class Tester:
    # INIT
    def __init__(self, data_test, batch_size, POS_WEIGHT_GENDER=1/2, POS_WEIGHT_BAG=1/2, POS_WEIGHT_HAT=1/2):
        self.device = device_selecter() # device selection

        self.data_test=data_test # dataset for test

        self.batch_size=batch_size

        self.losses_hat = [] 
        self.losses_gender = [] 
        self.losses_bag = [] 

        self.accuracies_hat = []
        self.accuracies_gender = [] 
        self.accuracies_bag = []

        self.val_losses_hat = []
        self.val_losses_gender = []
        self.val_losses_bag = [] 
        self.val_losses_tot = []

        self.val_accuracies_gender = [] 
        self.val_accuracies_bag = [] 
        self.val_accuracies_hat = [] 
        self.val_accuracies_tot = [] 

        self.f1_gender = [] 
        self.f1_bag = [] 
        self.f1_hat = [] 
        self.f1_tot = [] 

        self.POS_WEIGHT_GENDER=POS_WEIGHT_GENDER
        self.POS_WEIGHT_BAG=POS_WEIGHT_BAG
        self.POS_WEIGHT_HAT=POS_WEIGHT_HAT


    
    def tpfpfn(self, pred,labels):
        tp = ((pred == 1) & (labels.unsqueeze(1) == 1)).sum().item()
        fp = ((pred == 1) & (labels.unsqueeze(1) == 0)).sum().item()
        fn = ((pred == 0) & (labels.unsqueeze(1) == 1)).sum().item()
        return tp,fp,fn
    
    def fscore(self,tp,fp,fn):
        precision = tp / (tp + fp + 1e-8)
        recall_gender = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall_gender) / (precision + recall_gender + 1e-8)
        return f1


    def test(self,model,gender_weight=1/3,bag_weight=1/3,hat_weight=1/3):
        # model = CNNWithAttention() 
        # model.to(self.device)
        # checkpoint = torch.load(model_path)
        # model.load_state_dict(checkpoint['model_state_dict'])


        model.eval()  # Imposta il modello in modalità valutazione
        val_loss = 0.0
        val_acc_gender = 0.0
        val_acc_hat = 0.0
        val_acc_bag = 0.0
        total_samples = 0
        loss_val = 0
        tp_gender_tot = 0
        fp_gender_tot = 0
        fn_gender_tot = 0
        tp_hat_tot = 0
        fp_hat_tot = 0
        fn_hat_tot = 0
        tp_bag_tot = 0
        fp_bag_tot = 0
        fn_bag_tot = 0

        with torch.no_grad():  # Disabilita i gradienti
            for i, (images, labels) in enumerate(self.data_test):
                images = images.to(self.device)
                labels = labels.to(self.device)
                gender, bag, hat  = model(images)

                # Calcola le perdite
                loss_gender = adjustedLoss(gender, labels[:, 0], pos_weight= self.POS_WEIGHT_GENDER)
                loss_bag = adjustedLoss(bag, labels[:, 1], pos_weight= self.POS_WEIGHT_HAT)
                loss_hat = adjustedLoss(hat, labels[:, 2], pos_weight= self.POS_WEIGHT_BAG)

                        #loss_val = gradnorm_loss(loss_gender, loss_hat, loss_bag)
                loss_val = total_loss_fuction(loss_gender,loss_bag,loss_hat, gender_weight,  bag_weight, hat_weight)
                        
                val_loss += loss_val.item()

                        #Calcola l'accuratezza per ciascun output
                gender_pred = torch.sigmoid(gender) > 0.5
                accuracy_gender = (gender_pred.float() == labels[:, 0].unsqueeze(1)).float().mean()
                tp_gender, fp_gender, fn_gender = self.tpfpfn(gender_pred,labels[:,0])
                fgender = self.fscore(tp_gender, fp_gender, fn_gender)

                bag_pred = torch.sigmoid(bag) > 0.5
                accuracy_bag = (bag_pred.float() == labels[:, 1].unsqueeze(1)).float().mean()
                tp_bag, fp_bag, fn_bag = self.tpfpfn(bag_pred,labels[:,1])
                fbag = self.fscore(tp_bag, fp_bag, fn_bag)

                hat_pred = torch.sigmoid(hat) > 0.5
                accuracy_hat = (hat_pred.float() == labels[:, 2].unsqueeze(1)).float().mean()
                tp_hat, fp_hat, fn_hat = self.tpfpfn(hat_pred,labels[:,2])
                fhat = self.fscore(tp_hat, fp_hat, fn_hat)


                val_acc_gender += accuracy_gender.item()
                val_acc_hat += accuracy_hat.item()
                val_acc_bag += accuracy_bag.item()

                tp_gender_tot += tp_gender
                fp_gender_tot += fp_gender
                fn_gender_tot += fn_gender

                tp_hat_tot += tp_hat
                fp_hat_tot += fp_hat
                fn_hat_tot += fn_hat

                tp_bag_tot += tp_bag
                fp_bag_tot += fp_bag
                fn_bag_tot += fn_bag

                total_samples += 1
                print(f"Validation batch {i+1}/{len(self.data_test)}")

        # Calcola le medie
        val_loss /= total_samples
        val_acc_gender /= total_samples
        val_acc_hat /= total_samples
        val_acc_bag /= total_samples


        fgender = self.fscore(tp_gender_tot, fp_gender_tot, fn_gender_tot)
        fhat = self.fscore(tp_hat_tot, fp_hat_tot, fn_hat_tot)
        fbag = self.fscore(tp_bag_tot, fp_bag_tot, fn_bag_tot)

        # Salvo i valori di validazione per ogni epoca
        self.val_losses_tot.append(val_loss)
        self.val_accuracies_gender.append(val_acc_gender)
        self.val_accuracies_hat.append(val_acc_hat)
        self.val_accuracies_bag.append(val_acc_bag)

        self.f1_tot.append((fgender+fhat+fbag)/3)
        self.f1_gender.append(fgender)
        self.f1_hat.append(fhat)
        self.f1_bag.append(fbag)

        print(f" Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy (Gender): {val_acc_gender:.4f}")
        print(f"Validation Accuracy (Hat): {val_acc_hat:.4f}, Validation Accuracy (Bag): {val_acc_bag:.4f}")
        print(f"Tp (Gender): {tp_gender_tot:.4f}, Fp (Gender): {fp_gender_tot:.4f}, Fn (Gender): {fn_gender_tot}")
        print(f"Tp (Hat): {tp_hat_tot:.4f}, Fp (Hat): {fp_hat_tot:.4f}, Fn (hat): {fn_hat_tot}")
        print(f"Tp (Bag): {tp_bag_tot:.4f}, Fp (Bag): {fp_bag_tot:.4f}, Fn (Bag): {fn_bag_tot}")
        print(f"Fscore (Gender): {fgender:.2f}, Fscore (Hat): {fhat:.2f}, Fbag (Bag): {fbag}")
        print("Total Validation Samples: ", len(self.data_test) * self.batch_size)




    def plot(self, plots_name=None):

        if plots_name is None:
            plots_name = 'try'

        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(self.val_losses_tot) + 1), self.val_losses_tot, label='Validation Loss', marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Validation Loss')
        plt.grid(True)
        plt.legend()
        plt.savefig('./src/Classifier/Plots/validation_loss_'+ plots_name+'.png')


        # Plot della Validation Accuracy per Gender
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(self.val_accuracies_gender) + 1), self.val_accuracies_gender, label='Accuracy (Gender)', marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Validation Accuracy (Gender)')
        plt.grid(True)
        plt.legend()
        plt.savefig('./src/Classifier/Plots/accuracy_gender_'+ plots_name+'.png')


        # Plot della Validation Accuracy per Hat
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(self.val_accuracies_hat) + 1), self.val_accuracies_hat, label='Accuracy (Hat)', marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Validation Accuracy (Hat)')
        plt.grid(True)
        plt.legend()
        plt.savefig('./src/Classifier/Plots/accuracy_hat_'+ plots_name+'.png')


        # Plot della Validation Accuracy per Bag
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(self.val_accuracies_bag) + 1), self.val_accuracies_bag, label='Accuracy (Bag)', marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Validation Accuracy (Bag)')
        plt.grid(True)
        plt.legend()
        plt.savefig('./src/Classifier/Plots/accuracy_bag_'+ plots_name+'.png')

        # Plot del Validation F1 Score per Gender
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(self.f1_gender) + 1), self.f1_gender, label='F1 Score (Gender)', marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('F1 Score')
        plt.title('Validation F1 Score (Gender)')
        plt.grid(True)
        plt.legend()
        plt.savefig('./src/Classifier/Plots/f1_score_gender_' + plots_name + '.png')

        # Plot del Validation F1 Score per Hat
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(self.f1_hat) + 1), self.f1_hat, label='F1 Score (Hat)', marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('F1 Score')
        plt.title('Validation F1 Score (Hat)')
        plt.grid(True)
        plt.legend()
        plt.savefig('./src/Classifier/Plots/f1_score_hat_' + plots_name + '.png')

        # Plot del Validation F1 Score per Bag
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(self.f1_bag) + 1), self.f1_bag, label='F1 Score (Bag)', marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('F1 Score')
        plt.title('Validation F1 Score (Bag)')
        plt.grid(True)
        plt.legend()
        plt.savefig('./src/Classifier/Plots/f1_score_bag_' + plots_name + '.png')











