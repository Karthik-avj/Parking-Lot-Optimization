def f(xb,Nb,xd,Nd,w):
    
    xb_closed = list(xb.copy()[:2*Nb])
    xb_closed.append(xb[0])
    xb_closed.append(xb[1])
    xb_closed = np.array(xb_closed)
    xb_closed = np.reshape(xb_closed,(Nb+1,2))

    cost = 0;
    #Distance from boundary
    for i in range(Nb):
        if xb_closed[i,0]<w:
            cost = cost+1e20*(w-xb_closed[i,0])
        if xb_closed[i,1]<w:
            cost = cost+1e20*(w-xb_closed[i,1])
        if 1-xb_closed[i,0]<w:
            cost = cost+1e20*(xb_closed[i,0]-1+w)
        if 1-xb_closed[i,1]<w:
            cost = cost+1e20*(xb_closed[i,1]-1+w)
    
    #Distance from points in domain
    for i in range(Nd):
        v = np.zeros(Nb)
        dist0 = np.zeros(Nb)
        dmin = 1e20
        for j in range(Nb):
            #Minimum distance from each section of the road
            dist1 = dist(xb_closed[j,:], xb_closed[j+1,:],xd[i,:])
            dist2 = dist(xb_closed[j+1,:], xb_closed[j,:],xd[i,:])
            dist = max(dist1,dist2)
            dmin = min(dmin,dist)
            if dist<=w:
                v[j] = 1
                dist0[j] = dist

        #penalty if 2 or more non consecutive road segments are close to same point
        if len(np.where(np.diff(v)))>2:
            cost = cost+(w-np.sum(dist0(v==1))/np.sum(v))/w
        #penalty if no road segment is close to a point
        if dmin>w:
            cost = cost+(dmin-w)/w
    
    #penalty for sharp turns
    cb = 0.5*(1+cosangle(xb_closed[Nb,:],xb_closed[0,:],xb_closed[1,:]))
    if cb>0.5:
        cost = cost+5*(cb-0.5)/0.5
    for i in range(1,Nb):
        cb = 0.5*(1+cosangle(xb_closed[i-1,:],xb_closed[i,:],xb_closed[i+1,:]))
        if cb>0.5:
            cost = cost+5*(cb-0.5)/0.5
    return cost

def dist(x1,x2,x3):
    #Least distance of x3 from line segment x12
    d = 0
    x32 = x3-x2
    x12 = x1-x2
    l32 = np.linalg.norm(x32)
    l12 = np.linalg.norm(x12);
    if (l32 == 0) or (l12 == 0):
        d = 0
    else:
        cosx2 = np.dot(x32,x12)/(l32*l12)
    if cosx2<=0:
        d = l32
    else:
        d = l32*np.sqrt(1-cosx2**2)
    return d

def cosangle(x1,x2,x3):
    #angle between 2 vectors
    x12 = x1-x2
    x32 = x3-x2
    cosx2 = np.dot(x12,x32)/(np.linalg.norm(x12)*np.linalg.norm(x32))
    return cosx2

def circle_points(c,r,n):
    return [[c[0]+np.cos(2*pi/n*x)*r,c[1]+np.sin(2*pi/n*x)*r] for x in range(0,n+1)]


xd = np.random.random((100,2))
Nd = 100
w = 0.1

xcir = circle_points([0.5,0.5],0.4,20)[:20]
Nr = 20

x_sq = scipy.optimize.minimize(cost, xcir, args=(Nr,xd2,Nd,w), method='Nelder-Mead')

xf = np.reshape(x_2nd,(Nr,2))

xf1 = list(xf)
xf1.append(xf1[0])
xc1 = list(xcir)
xc1.append(xc1[0])

plt.plot(np.array(xf1)[:,0],np.array(xf1)[:,1],'-o',label='Final path')
plt.plot(np.array(xc1)[:,0],np.array(xc1)[:,1],'-o',label='Original path')
plt.plot(xd1[:,0],xd1[:,1],'.',label='Parking spots')
plt.legend(bbox_to_anchor=(0.3, 1.05))
plt.show()