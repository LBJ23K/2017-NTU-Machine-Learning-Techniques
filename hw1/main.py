import numpy as np
from sklearn import svm
import matplotlib.pyplot as pl
import math
from sklearn.utils import shuffle
import cvxopt

TRAIN_FILE = "features.train"
TEST_FILE = "features.test"


def printQuestion(question):
    print("\n############%s############" % question)


def q1():
    printQuestion("Q1")
    x = np.array([[1, 0], [0, 1], [0, -1], [-1, 0], [0, 2], [0, -2], [-2, 0]])
    y = np.array([-1, -1, -1, 1, 1, 1, 1])
    z = np.apply_along_axis(lambda x: np.array(
        [2 * x[1]**2 - 4 * x[0] + 1, x[0]**2 - 2 * x[1] - 3]), 1, x)
    clf = svm.SVC(kernel='linear', C=1e10)
    clf.fit(z, y)
    w = clf.coef_
    b = clf.intercept_
    print('w:', w)
    print('b:', b)
    # w = (1, 0), b = -4, hyperplane: z1 = 4
    pl.scatter(z[:, 0], z[:, 1], c=y)
    pl.vlines(abs(int(b[0])), np.amin(z[:, 1]), np.amax(z[:, 1]))
    pl.xlabel("z1")
    pl.ylabel("z2")
    pl.savefig('img/q1.png', dpi=100)
    pl.show()


def q2():
    printQuestion("Q2")
    x = np.array([[1, 0], [0, 1], [0, -1], [-1, 0], [0, 2], [0, -2], [-2, 0]])
    y = np.array([-1, -1, -1, 1, 1, 1, 1])
    n_samples = x.shape[0]
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K[i, j] = custom_kernel(x[i], x[j])
    # Q
    P = cvxopt.matrix(np.outer(y, y) * K)
    # p
    q = cvxopt.matrix(np.ones(n_samples) * -1)

    A = cvxopt.matrix(y.astype(float), (1, n_samples))
    b = cvxopt.matrix(0.0)

    G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
    h = cvxopt.matrix(np.zeros(n_samples))

    # solve QP problem
    solution = cvxopt.solvers.qp(P, q, G, h, A, b)

    # Lagrange multipliers
    alpha = np.ravel(solution['x'])
    print("alpha:", alpha)
    # Support vectors have non zero lagrange multipliers
    # sv = a > 1e-5
    sv_confition = alpha > 1e-5
    ind = np.arange(len(alpha))[sv_confition]
    sv_a = alpha[sv_confition]
    sv = x[sv_confition]
    sv_y = y[sv_confition]
    print("support vector:", sv)

    # Intercept, using first sv to calculate
    b = sv_y[0] - np.sum(sv_a * sv_y * K[ind[0]][ind])
    print("b:", b)


def custom_kernel(x1, x2):
    return (2 + np.dot(x1, x2)) ** 2


def loadData(filename):
    dataF = open(filename)
    matrix = np.array([[float(item) for item in line.split()]
                       for line in dataF.readlines()])
    x = matrix[:, 1:]
    y = matrix[:, :1].astype(int)
    return x, y


def q11():
    printQuestion("Q11")
    x_train, y_train = loadData(TRAIN_FILE)
    y_train_0 = (y_train == 0)
    y_train_0 = np.ravel(y_train_0)
    wAry = []
    cAry = [-5, -3, -1, 1, 3]
    for c in cAry:
        clf = svm.SVC(C=math.pow(10, c), kernel='linear', shrinking=False)
        print("C:", c)
        clf.fit(x_train, y_train_0)
        w = clf.coef_[0]
        b = clf.intercept_[0]
        wAry.append(np.linalg.norm(w))
    pl.scatter(cAry, wAry)
    pl.xlabel("log10 C")
    pl.ylabel("||w||")
    pl.savefig('img/q11.png', dpi=100)
    pl.show()


def q12():
    printQuestion("Q12")
    x_train, y_train = loadData(TRAIN_FILE)
    y_train_8 = (y_train == 8)
    y_train_8 = np.ravel(y_train_8)
    cAry = [-5, -3, -1, 1, 3]
    Ein_ary = []
    for c in cAry:
        print("C:", c)
        clf = svm.SVC(C=math.pow(10, c), kernel='poly', degree=2,
                      gamma=1, coef0=1)
        clf.fit(x_train, y_train_8)
        y_test = clf.predict(x_train)

        E_in = np.average(y_train_8 != y_test)
        Ein_ary.append(E_in)
    pl.scatter(cAry, Ein_ary)
    pl.xlabel("log10 C")
    pl.ylabel("Ein")
    pl.savefig('img/q12.png', dpi=100)
    pl.show()


def q13():
    printQuestion("Q13")
    x_train, y_train = loadData(TRAIN_FILE)
    y_train_8 = (y_train == 8)
    y_train_8 = np.ravel(y_train_8)
    cAry = [-5, -3]
    n_support = []
    for c in cAry:
        print("C:", c)
        clf = svm.SVC(C=math.pow(10, c), kernel='poly', degree=2,
                      gamma=1, coef0=1)
        clf.fit(x_train, y_train_8)
        n_support.append(clf.n_support_)
    n_support = np.array(n_support)
    pl.scatter(cAry, n_support[:, 1], label='8')
    pl.scatter(cAry, n_support[:, 0], label='not 8')
    pl.xlabel("log10 C")
    pl.ylabel("n support")
    pl.legend()
    pl.savefig('img/q13.png', dpi=100)
    pl.show()


def q14():
    printQuestion("Q14")
    x_train, y_train = loadData(TRAIN_FILE)
    y_train_0 = (y_train == 0)
    y_train_0 = np.ravel(y_train_0)
    n_samples = x_train.shape[0]
    cAry = [-3, - 2, -1, 0, 1]
    wAry = []
    for c in cAry:
        print("C:", c)
        clf = svm.SVC(C=math.pow(10, c), kernel='rbf', gamma=80)
        clf.fit(x_train, y_train_0)
        # print(clf.decision_function(clf.support_vectors_[50:80]))
        # print(np.sum(clf.dual_coef_[0]*))
        alpha = clf.dual_coef_[0]
        sv_idx = clf.support_
        sv = clf.support_vectors_
        y_int = y_train_0.astype(int)
        w = 0
        for i in range(len(sv_idx)):
            if i % (len(sv_idx) / 5) == 0:
                print(len(sv_idx) / 5)
            m = sv_idx[i]
            for j in range(len(sv_idx)):
                n = sv_idx[j]
                k_mn = np.exp(-80 * np.sum(sv[i] - sv[j])**2)
                w += alpha[i] * alpha[j] * y_int[m] * y_int[n] * k_mn
        wAry.append(math.sqrt(w))
    print("w array:", wAry)
    # wAry = [1 / 0.2907111385298988, 1 / 2.907111385298987, 1 /
    # 24.04908083771491, 1 / 175.25973555546773, 1 / 1623.2288024437194]
    pl.scatter(cAry, wAry)
    pl.xlabel("log10 C")
    pl.ylabel("distance")
    pl.savefig('img/q14.png', dpi=100)
    pl.show()


def q15():
    printQuestion("Q15")
    x_train, y_train = loadData(TRAIN_FILE)
    y_train_0 = (y_train == 0)
    y_train_0 = np.ravel(y_train_0)

    x_test, y_test = loadData(TEST_FILE)
    y_test_0 = (y_test == 0)
    y_test_0 = np.ravel(y_test_0)
    gammas = [0, 1, 2, 3, 4]
    Eout_ary = []
    for gamma in gammas:
        print("gamma:", gamma)
        clf = svm.SVC(C=0.1, kernel='rbf', gamma=math.pow(10, gamma))
        clf.fit(x_train, y_train_0)
        y_test = clf.predict(x_test)
        E_out = np.average(y_test_0 != y_test)
        Eout_ary.append(E_out)
    pl.scatter(gammas, Eout_ary)
    pl.xlabel("log10 gamma")
    pl.ylabel("Eout")
    pl.savefig('img/q15.png', dpi=100)
    pl.show()


def q16():
    printQuestion("Q16")
    x_train, y_train = loadData(TRAIN_FILE)
    y_train_0 = (y_train == 0)
    y_train_0 = np.ravel(y_train_0)
    gammas = [-1, 0, 1, 2, 3]
    count = [0] * len(gammas)
    for i in range(100):
        x, y = shuffle(x_train, y_train_0,
                       random_state=np.random.seed(i))
        Eval_ary = []
        print("count:", i)
        for gamma in gammas:
            clf = svm.SVC(C=0.1, kernel='rbf', gamma=math.pow(10, gamma))
            clf.fit(x[1000:], y[1000:])
            y_test = clf.predict(x[:1000])
            E_val = np.average(y_test != y[:1000])
            Eval_ary.append(E_val)
        count[np.argmin(Eval_ary)] += 1
    print("count array:", count)
    pl.bar(gammas, count, width=0.35)
    pl.xlabel("log10 gamma")
    pl.ylabel("number")
    pl.savefig('img/q16.png', dpi=100)
    pl.show()

if __name__ == "__main__":
    q1()
    q2()
    q11()
    q12()
    q13()
    q14()
    q15()
    q16()
