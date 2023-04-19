import numpy as np
import oqs
import tensorflow as tf
import jax.numpy as jnp
import jax
import optax
import random
import tensorcircuit as tc
from tqdm import tqdm
from hashlib import sha256

import time
from datetime import datetime
import os

date_time = datetime.now().strftime("%m%d%Y_%H%M%S")
logs = f"logs/{date_time}"

# os.mkdir(logs)

if not os.path.exists(logs):
		os.makedirs(logs)

sig_algs = ["Dilithium5","Falcon-1024","SPHINCS+-Haraka-256s-robust"]

with_scheme = True
# with_scheme = False
sigAlg = sig_algs[2]

expNo = 0

n = 8
n_node = 8
device_set = {}
n_class = 3

k = 12
readout_mode = 'softmax'
K = tc.set_backend('jax')  #set backend K
key = jax.random.PRNGKey(42)
tf.random.set_seed(42)


def filter(x, y, class_list):
    # print("Filter Method...")
    # print("x and y", x, y)
    keep = jnp.zeros(len(y)).astype(bool)
    # print("Keep", keep)
    for c in class_list:
        print("c", c)
        keep = keep | (y == c)
    x, y = x[keep], y[keep]
    y = jax.nn.one_hot(y, n_node)
    return x, y

'''The clf function in tensorcircuit is a quantum circuit that applies a series of gates to a given input circuit c, 
using a set of parameters params and a specified number of layers k. The purpose of this function 
is to implement a simple classifier using a quantum circuit.
'''

def clf(params, c, k):  #Quantum circuit that applies series of gates to a given input "c", using set of parameters from "params" and specified number of layers 'k'.
    for j in range(k):
        for i in range(n - 1):  #n is number of qubits in the input circuit 'c'
            c.cnot(i, i + 1)  #CNOT gate applied to adjacent qubits, flipping the second qubit if the first qubit is in the state |1> [Used to entangle qubits and create complex quantum states.
        for i in range(n):   #set of parameterized gates rx, rz, rx , acts on qubit individually, perform arbitrary rotations on the qubits.
            c.rx(i, theta=params[3 * j, i])  #params[0:2,:]
            c.rz(i, theta=params[3 * j + 1, i])
            c.rx(i, theta=params[3 * j + 2, i])
    return c  #return updated circuit

def readout(c):
    if readout_mode == 'softmax':
        logits = []
        for i in range(n_node):
            logits.append(jnp.real(c.expectation([tc.gates.z(), [i,]])))
        logits = jnp.stack(logits, axis=-1) * 10
        probs = jax.nn.softmax(logits)
    elif readout_mode == 'sample':
        wf = jnp.abs(c.wavefunction()[:n_node])**2
        probs = wf / jnp.sum(wf)
    return probs

def loss(params, x, y, k):
    c = tc.Circuit(n, inputs=x)
    c = clf(params, c, k)
    probs = readout(c)
    return -jnp.mean(jnp.sum(y * jnp.log(probs + 1e-7), axis=-1))
loss = K.jit(loss, static_argnums=[3])

def accuracy(params, x, y, k):
    c = tc.Circuit(n, inputs=x)
    c = clf(params, c, k)
    probs = readout(c)
    return jnp.argmax(probs, axis=-1) == jnp.argmax(y, axis=-1)
accuracy = K.jit(accuracy, static_argnums=[3])

compute_loss = K.jit(K.vectorized_value_and_grad(loss, vectorized_argnums=[1, 2]), static_argnums=[3])
compute_accuracy = K.jit(K.vmap(accuracy, vectorized_argnums=[1, 2]), static_argnums=[3])


def pred(params, x, k):
    c = tc.Circuit(n, inputs=x)
    c = clf(params, c, k)
    probs = readout(c)
    return probs
pred = K.vmap(pred, vectorized_argnums=[1])

class Device:
    def __init__(self, id, data, params, opt_state,  server):
        self.id = id
        # self.train = train
        # self.test = test
        self.data = data
        self.params = params
        self.opt_state = opt_state
        self.server = server
        self.sk = None
        self.params_hash = None
        self.pk = None
        if with_scheme:
            self.generate_key()
        self.train_list = []
        self.train_loss = []
        self.signature = None
        self.hash_signature = None

    def generate_key(self):
        # signer = oqs.Signature("Dilithium5")
        signer = oqs.Signature(sigAlg)
        self.pk = signer.generate_keypair()
        self.sk = signer.export_secret_key()

    def sign_msg(self, msg):

        # signer = oqs.Signature("Dilithium5", self.sk)
        signer = oqs.Signature(sigAlg, self.sk)
        signature = signer.sign(msg)
        return signature

    def verify_msg(self, msg, signature, signer_public_key):
        # verifier = oqs.Signature("Dilithium5")
        verifier = oqs.Signature(sigAlg)
        is_valid = verifier.verify(msg, signature, signer_public_key)
        return is_valid

def prepareData(dataset, encoding_mode): #preparing datasets
    if dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()  #mnist dataset
    elif dataset == 'fashion':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()  #fashion dataset

    # Remove labels 8 and 9 from test and train set
    ind = y_test == 9
    x_test, y_test = x_test[~ind], y_test[~ind]
    ind = y_test == 8
    x_test, y_test = x_test[~ind], y_test[~ind]
    ind = y_train == 9
    x_train, y_train = x_train[~ind], y_train[~ind]
    ind = y_train == 8
    x_train, y_train = x_train[~ind], y_train[~ind]

    x_train = x_train / 255.0   #normalized dataset
    # choose encoding mode type
    if encoding_mode == 'vanilla':
        mean = 0
    elif encoding_mode == 'mean':
        mean = jnp.mean(x_train, axis=0)
    elif encoding_mode == 'half':
        mean = 0.5
    x_train = x_train - mean

    # Resize data size so that they can be fit into quantum circuit
    x_train = tf.image.resize(x_train[..., tf.newaxis], (int(2 ** (n / 2)), int(2 ** (n / 2)))).numpy()[..., 0].reshape(
        -1, 2 ** n)
    x_train = x_train / jnp.sqrt(jnp.sum(x_train ** 2, axis=-1, keepdims=True))

    x_test = x_test / 255.0  #do same with test set
    x_test = x_test - mean
    x_test = tf.image.resize(x_test[..., tf.newaxis], (int(2 ** (n / 2)), int(2 ** (n / 2)))).numpy()[..., 0].reshape(
        -1, 2 ** n)
    x_test = x_test / jnp.sqrt(jnp.sum(x_test ** 2, axis=-1, keepdims=True))
    y_test = jax.nn.one_hot(y_test, n_node)

    return x_train, y_train, x_test, y_test

# Generate a random number to select a server randomly
# randomNumber = random.randint(0, n_node) #between 0 and n_node

randomNumber = random.sample(range(0, n_node), 2) #get two unique random values

print("TASK 1: Generating devices...")
for node in range(n_node-1):
    x_train, y_train, x_test, y_test = prepareData("mnist", "vanilla")
    deviceId = node
    x_train_node, y_train_node = filter(x_train, y_train, [(node + i) % n_node for i in range(n_class)])
    data = tf.data.Dataset.from_tensor_slices((x_train_node, y_train_node)).batch(128)
    y_train_cat = np.argmax(y_train_node, axis=1)
    print(f"Device {node}:{y_train_cat}")

    if randomNumber[0] == node:
        server = 1
    elif randomNumber[1] == node:
        server = 2
    else:
        server = 0

    key, subkey = jax.random.split(key)
    params = jax.random.normal(subkey, (3 * k, n))
    opt = optax.adam(learning_rate=1e-2)
    opt_state = opt.init(params)
    a_device = Device(deviceId, data, params, opt_state, server)
    device_set[node] = a_device

devices_list = list(device_set.values())

def workerTask(device, local_epochs, b):
    print(f"Device {device.id} training start...")
    for epoch in tqdm(range(local_epochs), leave=False):
        for i, (x, y) in enumerate(device.data):
            x = x.numpy()
            y = y.numpy()
            loss_val, grad_val = compute_loss(device.params, x, y, k)
            updates, device.opt_state = opt.update(grad_val, device.opt_state, device.params)
            device.params = optax.apply_updates(device.params, updates)  #updating device parameters
            device.params_hash = int.from_bytes(sha256(str(device.params).encode('utf-8')).digest(), byteorder='big')

            loss_mean = jnp.mean(loss_val)

            if i % 20 == 0:
                acc = jnp.mean(compute_accuracy(device.params, x, y, k))
                tqdm.write(
                    f'world {b}, epoch {epoch}, {i}/{len(device.data)}: loss={loss_mean:.4f}, acc={acc:.4f}')
                with open(f"{logs}/train_results.txt", "a") as file:
                    file.write(f"Comm: {b} - Device {device.id} - train_loss: {loss_mean} - train_acc: {acc}\n")

        print(f"Device {device.id} training Epoch: {epoch} done...")
    print(f"Device {device.id} training ALL EPOCHS done...")
    print("Signing the parameters locally.")
    if with_scheme:
        device.signature = device.sign_msg(str(device.params).encode())


def device_training(local_epochs, b):
    print("Device Training Method.... ")
    for device in devices_list:
        workerTask(device, local_epochs, b) #All devices train
        # if device.server == 1:
        #     pass
        # else:
        #     workerTask(device, local_epochs, b)


def aggregatorTask():
    total_hash = None
    for device in devices_list:
    # if device.server == 0:
        total_hash += device.params_hash
    for device in devices_list:
        if device.server == 2:
            device.hash_signature = device.sign_msg(total_hash)


avg_params = None
def serverTask(b):
    params_list = []
    serverDevice = None

    print("First set server device.")
    for device in devices_list:
        if device.server == 1:
            serverDevice = device
    for device in devices_list:
        if not device.server == 1:
            if with_scheme:
                print("Verifying device parameters")
                if serverDevice.verify_msg(str(device.params).encode(), device.signature, device.pk):
                    print("Params verified.")
                    params_list.append(device.params)
                else:
                    print("Device params compromized. Not added to averaging!")
            else:
                params_list.append(device.params)

    avg_params = jnp.mean(jnp.stack(params_list, axis=0), axis=0)

    for device in devices_list:
        device.params = avg_params

    test_acc = jnp.mean(pred(avg_params, x_test[:1024], k).argmax(axis=-1) == y_test[:1024].argmax(axis=-1))
    test_loss = -jnp.mean(jnp.log(pred(avg_params, x_test[:1024], k)) * y_test[:1024])

    tqdm.write(f'world {b}: test acc={test_acc:.4f}, test loss={test_loss:.4f}')

    with open(f"{logs}/server_test_results.txt", "a") as file:
        file.write(f"Comm: {b} - test_loss: {test_loss} - test_acc: {test_acc}\n")


loss_list = []
acc_list = []

print("Start Communication Rounds")
# for b in range(100):
#     current_time = time.time_ns()
#     print("Communication Round: ", b)
#     print(f"COMM ROUND: {b} - Start training Device")
#     device_training(1, b)
#     print(f"COMM ROUND: {b} - Start Server Task")
#     serverTask(b)
#     final_time = time.time_ns() - current_time
#     with open(f"{logs}/comm_time.txt", "a") as file:
#         file.write(f"Comm: {b} - Time: {final_time}\n")

for b in range(10):
    current_time = time.time_ns()
    print("Communication Round: ", b)
    print(f"COMM ROUND: {b} - Start training Device")
    device_training(1, b)
    print(f"COMM ROUND: {b} - Start Server Task")
    serverTask(b)
    final_time = time.time_ns() - current_time
    with open(f"{logs}/comm_time.txt", "a") as file:
        file.write(f"Comm: {b} - Time: {final_time}\n")


