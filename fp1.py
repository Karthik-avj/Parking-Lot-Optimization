def cross_entropy(f, c, x, x0, rho, gamma, nw, nel, nf, var, cov_fac):
    xhist = [x0]
    fhist = [f(x,x0)]
    chist = [max(max(c(x0)[0],0),max(c(x0)[1],0))]
    mean = x0
    cov = var*np.identity(len(x0))
    for k in range(20):
        rho *= gamma
        s = np.random.multivariate_normal(mean,cov,nw)
        vals = np.array([f(x,s[i]) for i in range(nw)])
        cs = np.array([c(s[i]) for i in range(nw)])
        val1 = np.zeros_like(vals)
        for i in range(len(cs)):
            val1[i] = vals[i]+rho*np.sum(np.maximum(cs[i],np.zeros_like(cs[i]))**2)
        elite = val1.argsort()[:nel]
        el_vals = np.array([s[i] for i in elite])
        mean = np.mean(el_vals, axis=0)
        cov *= cov_fac
        xhist.append(mean)
        fhist.append(f(x, mean))
        chist.append(max(max(c(mean)[0],0),max(c(mean)[1],0)))
        nw = int(nw*nf)
    return xhist, fhist, chist

def f1(x,x1):
    s = 0
    for i in range(len(x)):
        s+=np.linalg.norm(x[i]-x1)
    return -s

def f3(x,x1):
    s = 0
    for i in range(len(x)):
        s+=1/np.linalg.norm(x[i]-x1)
    return s

def c1(x1):
    return [-x1[0],-x1[1],x1[0]-10,x1[1]-10]

def each_iter(x):
    xhist1, fhist1, chist1 = cross_entropy(f3, c1, x, 10*np.random.random(2), 100, 3, 50, 5, 1, 2, 0.5)
    x = list(x)
    x.append(xhist1[-1])
    x = np.array(x)
    return x

xhist1, fhist1, chist1 = cross_entropy(f1, c1, x, 10*np.random.random(2), 100, 3, 50, 5, 1, 2, 0.5)
xhist2, fhist2, chist2 = cross_entropy(f1, c1, x, 10*np.random.random(2), 100, 3, 50, 5, 1, 2, 0.5)
xhist3, fhist3, chist3 = cross_entropy(f1, c1, x, 10*np.random.random(2), 100, 3, 50, 5, 1, 2, 0.5)

xr = np.linspace(-1,11,41)
yr = np.linspace(-1,11,41)

X,Y = np.meshgrid(xr,yr)

F = np.zeros_like(X)
for i in range(len(X)):
    for j in range(len(Y)):
        F[i,j] = f1(x,[X[i,j],Y[i,j]])
plt.contour(xr,yr,F,levels=[-1000000,-160,-140,-120,-100,-80,10000])
# plt.contour(xr,yr,F,levels=[80,100,120,140,160,180])
plt.plot(x[:,0],x[:,1],'.')
plt.plot([xhist1[i][0] for i in range(len(xhist1))], [xhist1[i][1] for i in range(len(xhist1))])
plt.plot([xhist2[i][0] for i in range(len(xhist2))], [xhist2[i][1] for i in range(len(xhist2))])
plt.plot([xhist3[i][0] for i in range(len(xhist3))], [xhist3[i][1] for i in range(len(xhist3))])
plt.xlabel('x1')
plt.ylabel('x2')
# plt.clim(0,10)
# plt.colorbar()
plt.contour(xr,yr,np.array(c1([X,Y]))[0],levels=[0])
plt.contour(xr,yr,np.array(c1([X,Y]))[1],levels=[0])
plt.contour(xr,yr,np.array(c1([X,Y]))[2],levels=[0])
plt.contour(xr,yr,np.array(c1([X,Y]))[3],levels=[0])
plt.title('Cross-entropy')
plt.savefig('s1o3.png')
plt.show()

x = np.array([np.zeros(2)])
for i in range(100):
    x = each_iter(x)

plt.plot(x[:,0],x[:,1],'.')