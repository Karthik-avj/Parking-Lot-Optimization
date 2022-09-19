xd2 = []
for i in range(3):
    xd2.append([0.5-i*0.15-0.075,1-0.075])
    xd2.append([0.5-i*0.15-0.075,0.075])
    xd2.append([0.5+i*0.15+0.075,1-0.075])
    xd2.append([0.5+i*0.15+0.075,0.075])
    xd2.append([1-0.075,0.5-i*0.15-0.075])
    xd2.append([0.075,0.5-i*0.15-0.075])
    xd2.append([1-0.075,0.5+i*0.15+0.075])
    xd2.append([0.075,0.5+i*0.15+0.075])
xd2.append([0.5-0.075,0.5-0.075])
xd2.append([0.5+0.075,0.5-0.075])
xd2.append([0.5-0.075,0.5+0.075])
xd2.append([0.5+0.075,0.5+0.075])

xd2 = np.array(xd2)
plt.plot(xd2[:,0],xd2[:,1],'.')
Nd2 = len(xd2)
w = 0.1

xcir = PointsInCircum([0.5,0.5],0.4,20)[:20]
Nr = 20

x_sq = scipy.optimize.minimize(cost, xcir, args=(Nr,xd2,Nd2,w), method='Nelder-Mead')

xf = np.reshape(x_2nd,(Nr,2))

xf1 = list(xf)
xf1.append(xf1[0])
xc1 = list(xcir)
xc1.append(xc1[0])

plt.plot(np.array(xf1)[:,0],np.array(xf1)[:,1],'-o',label='Final path')
plt.plot(np.array(xc1)[:,0],np.array(xc1)[:,1],'-o',label='Original path')
plt.plot(xd2[:,0],xd2[:,1],'.',label='Parking spots')
plt.plot([0.2,0.8,0.8,0.2,0.2],[0.2,0.2,0.8,0.8,0.2],label='Ideal path')
plt.legend(bbox_to_anchor=(0.3, 1.05))
plt.show()