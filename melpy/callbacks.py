import numpy as np
import matplotlib.pyplot as plt
from .tensor import *

class Callback:
    """
    A base class for creating custom callbacks to extend the functionality of the training loop.

    Attributes
    ----------
    None

    Methods
    -------
    on_train_start(model : Sequential, *args, **kwargs)
        Called at the start of the training loop.
    on_train_end(model : Sequential, *args, **kwargs)
        Called at the end of the training loop.
    on_epoch_start(model : Sequential, *args, **kwargs)
        Called at the start of each epoch.
    on_epoch_end(model : Sequential, *args, **kwargs)
        Called at the end of each epoch.
    on_step_start(model : Sequential, *args, **kwargs)
        Called at the start of each step.
    on_step_end(model : Sequential, *args, **kwargs)
        Called at the end of each step.
    """

    def __init__(self):
        """
        Initializes the Callback object.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        pass

    def on_train_start(self, model, *args, **kwargs):
        """
        Called at the start of the training loop.

        This method can be overridden to perform actions at the beginning of the training loop.

        Parameters
        ----------
        model : Sequential
            The model being trained.
        *args
            Additional positional arguments.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        None
        """
        pass

    def on_train_end(self, model, *args, **kwargs):
        """
        Called at the end of the training loop.

        This method can be overridden to perform actions at the end of the training loop.

        Parameters
        ----------
        model : Sequential
            The model being trained.
        *args
            Additional positional arguments.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        None
        """
        pass

    def on_epoch_start(self, model, *args, **kwargs):
        """
        Called at the start of each epoch.

        This method can be overridden to perform actions at the beginning of each epoch.

        Parameters
        ----------
        model : Sequential
            The model being trained.
        *args
            Additional positional arguments.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        None
        """
        pass

    def on_epoch_end(self, model, *args, **kwargs):
        """
        Called at the end of each epoch.

        This method can be overridden to perform actions at the end of each epoch.

        Parameters
        ----------
        model : Sequential
            The model being trained.
        *args
            Additional positional arguments.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        None
        """
        pass

    def on_step_start(self, model, *args, **kwargs):
        """
        Called at the start of each step.

        Parameters
        ----------
        model : Sequential
            The model being trained.
        *args
            Additional positional arguments.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        None
        """
        pass

    def on_step_end(self, model, *args, **kwargs):
        """
        Called at the end of each step.

        Parameters
        ----------
        model : Sequential
            The model being trained.
        *args
            Additional positional arguments.
        **kwargs
            Additional keyword arguments.
        Returns
        -------
        None
        """

class LiveMetrics(Callback):
    """
    A class for visualizing live metrics during the training of a neural network.

    Attributes
    ----------
    mode : int
        The mode of live metrics.
        1: Display loss and accuracy evolution.
        2: Display decision boundary.
        3: Display both loss/accuracy evolution and decision boundary.
    f1 : int
        The index of the first feature to use for plotting the decision boundary.
    f2 : int
        The index of the second feature to use for plotting the decision boundary.
    row_select : str
        Determines whether to use a limited or full set of rows for plotting.
        "limited": Use a limited number of rows.
        "full": Use the full set of rows.

    Methods
    -------
    on_iteration_end(model : Sequential, figure : Figure)
        Updates the live metrics plot based on the current state of the model.
    """

    def __init__(self, mode=3, f1=0, f2=1, row_select="limited"):
        """
        Initializes the LiveMetrics object with the specified parameters.

        Parameters
        ----------
        mode : int, optional
            The mode of live metrics to display. Default is 3.
            1: Display loss and accuracy evolution.
            2: Display decision boundary.
            3: Display both loss/accuracy evolution and decision boundary.
        f1 : int, optional
            The index of the first feature to use for plotting the decision boundary. Default is 0.
        f2 : int, optional
            The index of the second feature to use for plotting the decision boundary. Default is 1.
        row_select : str, optional
            Determines whether to use a limited or full set of rows for plotting. Default is "limited".
            "limited": Use a limited number of rows.
            "full": Use the full set of rows.

        Raises
        ------
        ValueError
            - If the `mode` parameter is not one of (1, 2, 3).
            - If the `row_select` parameter is not one of ("limited", "full").
            - If the `f1` or `f2` parameters are not integers.
        """
        if not isinstance(row_select, str):
            raise ValueError("`row_select` must be a string.")
        if not isinstance(f1, int):
            raise ValueError("`f1` must be an integer.")
        if not isinstance(f2, int):
            raise ValueError("`f2` must be an integer.")
        if not isinstance(mode, int):
            raise ValueError("`mode` must be an integer.")
        if mode not in (1, 2, 3):
            raise ValueError("`mode` must be one of (1, 2, 3).")
        if row_select not in ("limited", "full"):
            raise ValueError("`row_select` must be one of ('limited', 'full').")

        self.mode = mode
        self.f1 = f1
        self.f2 = f2
        self.row_select = row_select

    def on_epoch_end(self, model, figure):
        """
        Updates the live metrics plot based on the current state of the model.

        This method updates the live metrics plot based on the current state of the model.
        It supports different types of live metrics visualizations, including loss and accuracy
        evolution, and decision boundary plots.

        Parameters
        ----------
        model : Sequential
            The model whose metrics are to be visualized.
        figure : Figure
            The figure object to update with the live metrics plot.

        Raises
        ------
        ValueError
            If the `row_select` parameter is not one of ("limited", "full").
        """
        if self.mode == 1:
            figure.clear()

            plt.subplot(1, 2, 1)
            plt.title("Loss Evolution")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.plot(model.train_loss_history, color="#1f77b4", label=f"training dataset: {round(float(model.train_loss_history[-1]), 3)}")
            if model.val_inputs is not None and model.val_targets is not None:
                plt.plot(model.val_loss_history, color="orange", label=f"validation dataset: {round(float(model.val_loss_history[-1]), 3)}")
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.title("Accuracy Evolution")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.plot(model.train_accuracy_history, color="#1f77b4", label=f"training dataset: {round(float(model.train_accuracy_history[-1]), 3)}")
            if model.val_inputs is not None and model.val_targets is not None:
                plt.plot(model.val_accuracy_history, color="orange", label=f"validation dataset: {round(float(model.val_accuracy_history[-1]), 3)}")
            plt.legend()

            figure.canvas.draw()
            figure.canvas.flush_events()

        elif self.mode == 2:
            figure.clear()

            plt.subplot(1, 1, 1)
            if len(model.train_input_batch.array[0, :]) <= 2 and len(model.train_targets.array[0, :]) == 1:
                if len(model.train_input_batch.array[:, 0]) > 1000 and self.row_select == "limited":
                    X_set, y_set = model.train_input_batch.array[:1000, :], model.train_targets.array[:1000, :]
                elif len(model.train_input_batch.array[:, 0]) > 1000 and self.row_select == "full":
                    X_set, y_set = model.train_input_batch, model.train_targets
                elif (len(model.train_input_batch.array[:, 0]) <= 1000 and self.row_select == "full") or (len(model.train_input_batch.array[:, 0]) <= 1000 and self.row_select == "limited"):
                    X_set, y_set = model.train_input_batch.array, model.train_targets.array

                X1, X2 = np.meshgrid(np.arange(start=X_set[:, self.f1].min() - 1, stop=X_set[:, self.f1].max() + 1, step=0.01),
                                     np.arange(start=X_set[:, self.f2].min() - 1, stop=X_set[:, self.f2].max() + 1, step=0.01))

                predictions = model.predict(Tensor(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape)

                plt.contourf(X1, X2, predictions,
                             alpha=0.3, cmap="coolwarm", c=predictions.ravel())

            if len(model.train_input_batch.array[:, 0]) > 1000 and self.row_select == "limited":
                plt.scatter(model.train_input_batch.array[:1000, self.f1], model.train_input_batch.array[:1000, self.f2], c=model.train_output_batch.array[:1000, :], cmap="coolwarm", alpha=1)
            elif len(model.train_input_batch.array[:, 0]) > 1000 and self.row_select == "full":
                plt.scatter(model.train_input_batch.array[:, self.f1], model.train_input_batch.array[:, self.f2], c=model.train_output_batch.array[:, :], cmap="coolwarm", alpha=1)
            elif (len(model.train_input_batch.array[:, 0]) <= 1000 and self.row_select == "full") or (len(model.train_input_batch.array[:, 0]) <= 1000 and self.row_select == "limited"):
                plt.scatter(model.train_input_batch.array[:, self.f1], model.train_input_batch.array[:, self.f2], c=model.train_output_batch[:, :], cmap="coolwarm", alpha=1)

            figure.canvas.draw()
            figure.canvas.flush_events()

        elif self.mode == 3:
            figure.clear()

            plt.subplot(1, 2, 2)
            if len(model.train_input_batch.array[0, :]) <= 2 and len(model.train_targets.array[0, :]) == 1:
                if len(model.train_input_batch.array[:, 0]) > 1000 and self.row_select == "limited":
                    X_set, y_set = model.train_input_batch.array[:1000, :], model.train_targets.array[:1000, :]
                elif len(model.train_input_batch.array[:, 0]) > 1000 and self.row_select == "full":
                    X_set, y_set = model.train_input_batch.array, model.train_targets.array
                elif (len(model.train_input_batch.array[:, 0]) <= 1000 and self.row_select == "full") or (len(model.train_input_batch.array[:, 0]) <= 1000 and self.row_select == "limited"):
                    X_set, y_set = model.train_input_batch.array, model.train_targets.array

                X1, X2 = np.meshgrid(np.arange(start=X_set[:, self.f1].min() - 1, stop=X_set[:, self.f1].max() + 1, step=0.01),
                                     np.arange(start=X_set[:, self.f2].min() - 1, stop=X_set[:, self.f2].max() + 1, step=0.01))

                predictions =  model.predict(Tensor(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape)

                plt.contourf(X1, X2, predictions,
                             alpha=0.3, cmap="coolwarm", c=predictions.ravel())

            if len(model.train_input_batch.array[:, 0]) > 1000 and self.row_select == "limited":
                plt.scatter(model.train_input_batch.array[:1000, self.f1], model.train_input_batch.array[:1000, self.f2], c=model.train_output_batch.array[:1000, :], cmap="coolwarm", alpha=1)
            elif len(model.train_input_batch.array[:, 0]) > 1000 and self.row_select == "full":
                plt.scatter(model.train_input_batch.array[:, self.f1], model.train_input_batch.array[:, self.f2], c=model.train_output_batch.array[:, :], cmap="coolwarm", alpha=1)
            elif (len(model.train_input_batch.array[:, 0]) <= 1000 and self.row_select == "full") or (len(model.train_input_batch.array[:, 0]) <= 1000 and self.row_select == "limited"):
                plt.scatter(model.train_input_batch.array[:, self.f1], model.train_input_batch.array[:, self.f2], c=model.train_output_batch.array[:, :], cmap="coolwarm", alpha=1)

            plt.subplot(2, 2, 1)
            plt.title("Loss Evolution")
            plt.ylabel("Loss")
            plt.plot(model.train_loss_history, color="#1f77b4", label=f"training dataset: {round(float(model.train_loss_history[-1]), 3)}")
            if model.val_inputs is not None and model.val_targets is not None:
                plt.plot(model.val_loss_history, color="orange", label=f"validation dataset: {round(float(model.val_loss_history[-1]), 3)}")
            plt.legend()

            plt.subplot(2, 2, 3)
            plt.title("Accuracy Evolution")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.plot(model.train_accuracy_history, color="#1f77b4", label=f"training dataset: {round(float(model.train_accuracy_history[-1]), 3)}")
            if model.val_inputs is not None and model.val_targets is not None:
                plt.plot(model.val_accuracy_history, color="orange", label=f"validation dataset: {round(float(model.val_accuracy_history[-1]), 3)}")
            plt.legend()

            figure.canvas.draw()
            figure.canvas.flush_events()