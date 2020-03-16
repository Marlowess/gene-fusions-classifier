from utils.models.model_utils.plot_utils.plot_functions import *
from pprint import pprint

def plot_loss_and_accuracy(model, x_test, y_test, history, epochs: int, directory_figures: str, conf_images: dict = None) -> None:
    """
    This function creates and saves the plots about loss and accuracy

    Params:
    -------
        :model: The model used for testing data\n
        :x_test: Test samples' features\n
        :y_test: Test samples' labels\n
        :history: A dictionary containing loss and accuracy variation\n
        :epochs: Duration of the training in epochs\n
        :directory_figures: Path where to save figures\n
        :conf_images: Settings of images to create\n
    """
    
    print("\n", "-" * 100, sep="")
    
    # Loss Image
    pprint(conf_images['loss'])
    
    conf_loss_img: dict = conf_images['loss']
    savefig_flag: bool = conf_loss_img['savefig_flag']
    fig_name: str = conf_loss_img['fig_name']
    title: str = conf_loss_img['title']
    fig_format: str = conf_loss_img['fig_format']
    
    plot_loss(
        history=history,
        epochs=epochs,
        fig_dir=directory_figures,
        fig_name=fig_name,
        title=title,
        fig_format=fig_format,
        savefig_flag=savefig_flag)
    
    # Acc Image
    pprint(conf_images['acc'])
    
    conf_acc_img: dict = conf_images['acc']
    savefig_flag: bool = conf_acc_img['savefig_flag']
    fig_name: str = conf_acc_img['fig_name']
    title: str = conf_acc_img['title']
    fig_format: str = conf_acc_img['fig_format']
    
    plot_accuracy(
        history=history,
        epochs=epochs,
        fig_dir=directory_figures,
        fig_name=fig_name,
        title=title,
        fig_format=fig_format,
        savefig_flag=savefig_flag)
    pass

def plot_roc_curve_model(model, x_test, y_test, directory_figures: str, conf_images: dict, postfix: str = 'Train'):

    # fig_name : can NOT be empty since plot_roc_curce requires a not empty fig_name.
    """
    This function creates and saves the ROC curve plot

    Params:
    -------
        :model: The model used for testing data\n
        :x_test: Test samples' features\n
        :y_test: Test samples' labels\n
        :directory_figures: Path where to save figures\n
        :conf_images: Settings of images to create\n
        :postfix: Just a string used for the plot's title\n
    """

    print("\n", "-" * 100, sep="")
    
    pprint(conf_images['roc_curve'])
    roc_curve_conf: dict = conf_images['roc_curve']
    
    savefig_flag: bool = roc_curve_conf['savefig_flag']
    fig_name: str = roc_curve_conf['fig_name'] + '_' + postfix.lower()
    title: str = roc_curve_conf['title'] + ' ' + postfix
    
    if 'fig_format' not in roc_curve_conf.keys():
        fig_format: str = 'png'
    else:
        fig_format: str = roc_curve_conf['fig_format']
    
    y_pred = model.predict(x_test).ravel()
     
    auc = plot_roc_curce(
        y_test, y_pred,
        fig_dir=directory_figures,
        fig_name=fig_name,
        title=title,
        fig_format=fig_format,
        savefig_flag=savefig_flag)

    return auc
    
def plot_confusion_matrix_model(model, x_test, y_test, directory_figures: str, conf_images: dict, postfix: str = 'Train', sequential_model_flag: bool = True):
    
    # fig_name : can NOT be empty since plot_confusion_matrix requires a not empty fig_name.
    """
    This function creates and saves the confusion matrix of the model

    Params:
    -------
        :model: The model used for testing data\n
        :x_test: Test samples' features\n
        :y_test: Test samples' labels\n
        :directory_figures: Path where to save figures\n
        :conf_images: Settings of images to create\n
        :postfix: Just a string used for the plot's title\n
    """

    print("\n", "-" * 100, sep="")

    class_names = np.array(['Honcogenic', 'Non-Honcogenic'])
    
    pprint(conf_images['confusion_matrix'])
    confusion_matrix_conf: dict = conf_images['confusion_matrix']
    
    savefig_flag: bool = confusion_matrix_conf['savefig_flag']
    fig_name: str = confusion_matrix_conf['fig_name'] + '_' + postfix.lower()
    title: str = confusion_matrix_conf['title'] + ' ' + postfix
    
    if 'fig_format' not in confusion_matrix_conf.keys():
        fig_format: str = 'png'
    else:
        fig_format: str = confusion_matrix_conf['fig_format']
    
    if sequential_model_flag is True:
        pred = model.predict(x_test)
        y_pred_class = list(map(lambda x: 1 if x >= 0.50 else 0, pred))
    else:
        y_pred_class = model.predict_classes(x_test)
    
    cm, cm_norm = plot_confusion_matrix(
            y_test, y_pred_class,
            class_names,
            fig_dir=directory_figures,
            fig_name=fig_name,
            title=title,
            fig_format=fig_format,
            savefig_flag=savefig_flag)
    
    return cm, cm_norm
