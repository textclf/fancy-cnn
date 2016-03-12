from keras.callbacks import EarlyStopping, ModelCheckpoint
from time import strftime
import sys

def train_sequential(model, X, y, where_to_save, fit_params=None, monitor='val_acc'):
    # TODO: DOCUMENT once thoroughly tested
    # Watch out: where_to_save might be inside fit_params

    if fit_params is None:
        fit_params = {
            "batch_size": 32,
            "nb_epoch": 45,
            "verbose": True,
            "validation_split": 0.15,
            "show_accuracy": True,
            "callbacks": [EarlyStopping(verbose=True, patience=5, monitor=monitor),
                          ModelCheckpoint(where_to_save, monitor=monitor, verbose=True, save_best_only=True)]
        }
    print 'Fitting! Hit CTRL-C to stop early...'
    history = "Nothing to show"
    try:
        history = model.fit(X, y, **fit_params)
    except KeyboardInterrupt:
        print "Training stopped early!"
        history = model.history

    return history

def train_graph(model, fit_params):
    # TODO: DOCUMENT once thoroughly tested

    print 'Fitting! Hit CTRL-C to stop early...'
    history = "Nothing to show"
    try:
        history = model.fit(**fit_params)
    except KeyboardInterrupt:
        history = model.history
        print "Training stopped early!"

    return history

def test_sequential(model_obj, X_test, y_test, saved_model):
    # TODO: DOCUMENT

    model_obj.load_weights(saved_model)

    print "getting predictions on the test set"
    yhat = model_obj.predict(X_test, verbose=True, batch_size=50)
    acc = ((yhat.ravel() > 0.5) == (y_test > 0.5)).mean()

    print "Test set accuracy of {}%.".format(acc * 100.0)
    print "Test set error of {}%. Exiting...".format((1 - acc) * 100.0)

    return acc

def test_graph(model_obj, X_test, output_name, y_test,saved_model):
    # TODO: DOCUMENT

    model_obj.load_weights(saved_model)

    print "getting predictions on the test set"
    yhat = model_obj.predict(X_test, verbose=True, batch_size=50)
    acc = ((yhat[output_name].ravel() > 0.5) == (y_test > 0.5)).mean()

    print "Test set accuracy of {}%.".format(acc * 100.0)
    print "Test set error of {}%. Exiting...".format((1 - acc) * 100.0)

    return acc

def write_log(model, history, code_file, acc, log_file):

    def print_history(history_obj):
        for (i, (loss, val_loss)) in enumerate(zip(history_obj.history['loss'],
                                                   history_obj.history['val_loss'])):
            print "Epoch %d: loss: %f, val_loss: %f" % (i+1, loss, val_loss)

    # TODO: A bit tacky...watch out the sys.stdout thing...the terminal might disappear!!
    sys.stdout = open(log_file, 'w')

    print ("Model trained at " + strftime("%Y-%m-%d %H:%M:%S"))
    print ("Accuracy obtained: " + str(acc))
    print ("Error obtained: " + str(1 - acc))
    print ("==" * 40)
    print ("Model in json:")
    print ("==" * 40)
    try:
        print model.to_json()
    except Exception:
        print ('Error in model JSON encoding')
    print ("==" * 40)
    print "Model summary:"
    print ("==" * 40)
    try:
        model.summary()
    except Exception:
        print ('Error in printing model summary (you probably have a model with a Merge() layer somewhere.)')
    print ("==" * 40)
    print ("Training history:")
    print ("==" * 40)
    print_history(history)
    print ("==" * 40)
    print ("Code file:")
    print ("==" * 40)
    with open(code_file) as code:
        print (code.read())

    sys.stdout = sys.__stdout__
