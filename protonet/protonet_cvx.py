import numpy as np
import cvxpy as cp

data = np.load('protonet2.npz')

labels_query = data['labels_query']
constraints_all = data['constraints_all']
distances_query = data['distances_query']
weight = data['weight']
bias = data['bias']

w_0 = np.concatenate((weight.reshape((weight.shape[0], -1)), bias[...,None]), axis=1)

print(labels_query.shape)
print(constraints_all.shape)
print(distances_query.shape)

num_tasks = labels_query.shape[0]
num_layers = distances_query.shape[-2]
num_weights = distances_query.shape[-1]
num_query = distances_query.shape[1]
num_classes = distances_query.shape[2]

w = cp.Variable((num_layers, num_weights))

constraints = []

objective = 0

for i in range(num_layers):
    constraints.append(constraints_all[:, :, i].reshape(-1, num_weights) @ w[i] >= 0)

for i in range(num_tasks):
    print(f'task {i}')
    for j in range(num_query):
        print(f'query {j}')

        for k in range(num_layers):

            terms = []

            for l in range(num_classes):

                A = distances_query[i, j, l, :, k]

                terms.append((-2 * (A.T @ (A @ w_0[i]))) @ (w[i] - w_0[i])) # grad_w (|Ax|_2^2) = 2 * A.T * A * x

                if l == labels_query[i, j]: # if this is the class then add the sum of squares
                    objective += cp.sum_squares(A @ w[i])

            objective += cp.log_sum_exp(cp.hstack(terms))

objective += 1e-4*cp.mixed_norm(w, 2, 1)

print('finished building!')

prob = cp.Problem(cp.Minimize(objective))#, constraints)
prob.solve(verbose=True)
