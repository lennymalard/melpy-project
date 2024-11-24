import numpy as np
import matplotlib.pyplot as plt

class LiveMetrics():
    def __init__(self, type=-1, f1=0, f2=1, row_select="limited"):
        super().__init__()
        self.type = type
        self.f1 = f1
        self.f2 = f2
        self.row_select = row_select
        
        if self.type not in [-1,0,1,2]:
            raise ValueError("invalid value for 'type'")

    def run(self, model, figure):
        
        if self.type == 1:
            figure.clear()
            
            plt.subplot(1,2,1)
            plt.title("Loss Evolution")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.plot(model.train_loss_history, color="#1f77b4", label=f"training dataset: {round(float(model.train_loss_history[-1]), 3)}")
            if model.val_inputs is not None and model.val_targets is not None:
                plt.plot(model.val_loss_history, color="orange", label=f"validation dataset: {round(float(model.val_loss_history[-1]), 3)}")
            plt.legend()
            
            plt.subplot(1,2,2)
            plt.title("Accuracy Evolution")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.plot(model.train_accuracy_history, color="#1f77b4", label=f"training dataset: {round(float(model.train_accuracy_history[-1]), 3)}")
            if model.val_inputs is not None and model.val_targets is not None:
                plt.plot(model.val_accuracy_history, color="orange", label=f"validation dataset: {round(float(model.val_accuracy_history[-1]), 3)}")
            plt.legend()
            
            figure.canvas.draw()
            figure.canvas.flush_events()
            
        elif self.type == 2:
            figure.clear()
            
            plt.subplot(1,1,1)
            if len(model.train_input_batch[0,:]) <= 2 and len(model.train_targets[0,:]) == 1:
                if len(model.train_input_batch[:,0]) > 1000 and self.row_select=="limited":
                    X_set, y_set = model.train_input_batch[:1000,:], model.train_targets[:1000,:]
                elif len(model.train_input_batch[:,0]) > 1000 and self.row_select=="full":
                    X_set, y_set = model.train_input_batch, model.train_targets
                elif (len(model.train_input_batch[:,0]) <= 1000 and self.row_select=="full") or (len(model.train_input_batch[:,0]) <= 1000 and self.row_select=="limited"):
                    X_set, y_set = model.train_input_batch, model.train_targets
                else:
                    raise ValueError("invalid value for 'row_select'")
                    
                X1, X2 = np.meshgrid(np.arange(start = X_set[:, self.f1].min() - 1, stop = X_set[:, self.f1].max() + 1, step = 0.01),
                                     np.arange(start = X_set[:, self.f2].min() - 1, stop = X_set[:, self.f2].max() + 1, step = 0.01))
                plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                             alpha = 0.3, cmap = "coolwarm")
            
            if len(model.train_input_batch[:,0]) > 1000 and self.row_select=="limited":
                plt.scatter(model.train_input_batch[:1000,self.f1], model.train_input_batch[:1000,self.f2], c=model.train_output_batch[:1000,:], cmap="coolwarm", alpha=1)
            elif len(model.train_input_batch[:,0]) > 1000 and self.row_select=="full":
                plt.scatter(model.train_input_batch[:,self.f1], model.train_input_batch[:,self.f2], c=model.train_output_batch[:,:], cmap="coolwarm", alpha=1)
            elif (len(model.train_input_batch[:,0]) <= 1000 and self.row_select=="full") or (len(model.train_input_batch[:,0]) <= 1000 and self.row_select=="limited"):
                plt.scatter(model.train_input_batch[:,self.f1], model.train_input_batch[:,self.f2], c=model.train_output_batch[:,:], cmap="coolwarm", alpha=1)
            else:
                raise ValueError("invalid value for 'row_select'")
                
            figure.canvas.draw()
            figure.canvas.flush_events()
            
        elif self.type == -1:
            figure.clear()
            
            plt.subplot(1, 2, 2)
            if len(model.train_input_batch[0,:]) <= 2 and len(model.train_targets[0,:]) == 1:
                if len(model.train_input_batch[:,0]) > 1000 and self.row_select=="limited":
                    X_set, y_set = model.train_input_batch[:1000,:], model.train_targets[:1000,:]
                elif len(model.train_input_batch[:,0]) > 1000 and self.row_select=="full":
                    X_set, y_set = model.train_input_batch, model.train_targets
                elif (len(model.train_input_batch[:,0]) <= 1000 and self.row_select=="full") or (len(model.train_input_batch[:,0]) <= 1000 and self.row_select=="limited"):
                    X_set, y_set = model.train_input_batch, model.train_targets
                else:
                    raise ValueError("invalid value for 'row_select'")
                X1, X2 = np.meshgrid(np.arange(start = X_set[:, self.f1].min() - 1, stop = X_set[:, self.f1].max() + 1, step = 0.01),
                                     np.arange(start = X_set[:, self.f2].min() - 1, stop = X_set[:, self.f2].max() + 1, step = 0.01))
                plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                             alpha = 0.3, cmap = "coolwarm")
                
            if len(model.train_input_batch[:,0]) > 1000 and self.row_select=="limited":
                plt.scatter(model.train_input_batch[:1000,self.f1], model.train_input_batch[:1000,self.f2], c=model.train_output_batch[:1000,:], cmap="coolwarm", alpha=1)
            elif len(model.train_input_batch[:,0]) > 1000 and self.row_select=="full":
                plt.scatter(model.train_input_batch[:,self.f1], model.train_input_batch[:,self.f2], c=model.train_output_batch[:,:], cmap="coolwarm", alpha=1)
            elif (len(model.train_input_batch[:,0]) <= 1000 and self.row_select=="full") or (len(model.train_input_batch[:,0]) <= 1000 and self.row_select=="limited"):
                plt.scatter(model.train_input_batch[:,self.f1], model.train_input_batch[:,self.f2], c=model.train_output_batch[:,:], cmap="coolwarm", alpha=1)
            else:
                raise ValueError("invalid value for 'row_select'")
            
            plt.subplot(2,2,1)
            plt.title("Loss Evolution")
            plt.ylabel("Loss")
            plt.plot(model.train_loss_history, color="#1f77b4", label=f"training dataset: {round(float(model.train_loss_history[-1]), 3)}")
            if model.val_inputs is not None and model.val_targets is not None:
                plt.plot(model.val_loss_history, color="orange", label=f"validation dataset: {round(float(model.val_loss_history[-1]), 3)}")
            plt.legend()
            
            plt.subplot(2,2,3)
            plt.title("Accuracy Evolution")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.plot(model.train_accuracy_history, color="#1f77b4", label=f"training dataset: {round(float(model.train_accuracy_history[-1]), 3)}")
            if model.val_inputs is not None and model.val_targets is not None:
                plt.plot(model.val_accuracy_history, color="orange", label=f"validation dataset: {round(float(model.val_accuracy_history[-1]), 3)}")
            plt.legend()
            
            figure.canvas.draw()
            figure.canvas.flush_events()