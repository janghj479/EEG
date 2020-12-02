import mne
import numpy as np
import matplotlib.pyplot as plt


data_path = 'C:/Users/oo/Desktop/EEG/Hypnosis/Data'
fname = data_path + '/Hypnosis_S1_POST_ACTIVE_a.set' 
data = mne.io.read_raw_eeglab(fname)

data.plot()

# events=mne.find_events(data)

events = mne.events_from_annotations(data) #소리 들려주는거
epochs = mne.Epochs(data,events[0],tmin=-0.2, tmax=0.5) #신호발생 전후 구간

print(epochs)
print(epochs.event_id)
event_dict = {'dev': 1, 'std': 2} #dev - 높은소리?,60개 std 기본소리,420개

epochs = mne.Epochs(data, events[0], tmin=-0.2, tmax=0.5, 
                    event_id=event_dict, preload=True)

del data #원래 데이터 삭제 -용량이 크니까

epochs.plot(n_epochs=10); 
#%%
print(epochs['dev']) 
print(epochs['std']) 
#evoked = epochs.average() # 480개 평균

evoked_dev = epochs['dev']
evoked_std = epochs['std']

evoked_dev = epochs['dev'].average()
evoked_std = epochs['std'].average()

#std.evoked = epochs[2].average()
#%%
fig1 = plt.figure();
evoked_dev = fig1.add_subplot(1, 2, 1)
evoked_std = fig1.add_subplot(1, 2, 2)

ax1.plot(evoked_dev,1,2,1);
plt.grid();
evoked_std.plot(122);
plt.grid();

# %%

dev=evoked_dev.data
std=evoked_std.data
time = evoked_dev.times

dif = dev-std
plt.plot(time,dif[0,:])
#len(epochs.ch_names)

mean_dev=np.mean(dev,0)
mean_std=np.mean(std,0)

plt.figure(figsize=(10,6))

plt.subplot(211)
plt.plot(time,dev.T)
plt.subplot(212)
plt.plot(time,std.T)
plt.show()

#%%
from mne.viz import plot_evoked_topo
from mne.datasets import sample

evokeds = [epochs[name].average() for name in ('dev', 'std')]

colors = 'blue', 'red'
title = 'MNE sample data\n dev vs std '

plot_evoked_topo(evokeds, color=colors, title=title, background_color='w')
mne.viz.plot_compare_evokeds(evokeds, picks='eeg', vlines=[0,0.3], colors=dict(dev='b', std='r'),
                              axes='topo',  styles=dict(dev=dict(linewidth=1),
                                                     std=dict(linewidth=1)))


mne.viz.plot_compare_evokeds(evokeds, picks='eeg', colors=dict(aud=0, vis=1),
                             linestyles=dict(left='solid', right='dashed'),
                             axes='topo', styles=dict(aud=dict(linewidth=1),
                                                      vis=dict(linewidth=1)))

# %%


"""
evokeds = [epochs[name].average() for name in ('dev', 'std')]
evokeds.plot('dev', 'std');
"""

"""
times = np.arange(-0.2,0.55,0.15) #

fig2 = plt.figure();
evoked_dev.plot_topomap(times) = fig2.add_subplot(1, 2, 1)
evoked_std.plot_topomap(times) = fig2.add_subplot(1, 2, 2)

evoked_dev.plot_topomap(times);
evoked_std.plot_topomap(times);

plt.show();

#evoked.plot_topomap(times) #머리에 뿌리기
"""

""" 
지금한거는 dev신호랑 std신호 모두 평균내서 나온거고
과제로 할거는 둘이 따로 topomap그리기
"""
#%% 20.11.25
import glob
import mne
import numpy as np
import matplotlib.pyplot as plt



data_list = glob.glob('C:/Users/oo/Desktop/EEG/Hypnosis/Data/*a.set')

for i in range(0,8):
    globals()['d{}'.format(i)]=mne.io.read_raw_eeglab(data_list[i])

d=[d0,d1,d2,d3,d4,d5,d6,d7]


for i in range(0,8):
    globals()['events{}'.format(i)] = mne.events_from_annotations(d[i])

events= [events0,events1,events2,events3,events4,events5,events6,events7]

for i in range(0,8):
    globals()['epochs{}'.format(i)] = mne.Epochs(d[i],(events[i])[0],tmin=-0.2, tmax=0.5) 

epochs= [epochs0,epochs1,epochs2,epochs3,epochs4,epochs5,epochs6,epochs7]

'''
print(epochs7,epochs5)
print(epochs7.event_id,epochs5.event_id)
'''

for i in range(0,8):
    globals()['event_dict{}'.format(i)] ={'dev': 1, 'std': 2} 
    
event_dict=[event_dict0,event_dict1,event_dict2,event_dict3,event_dict4,event_dict5,event_dict6,event_dict7]

for i in range(0,8):
    globals()['epochs{}'.format(i)] =mne.Epochs(d[i], (events[i])[0], tmin=-0.2, tmax=0.5, 
                    event_id=event_dict[i], preload=True)
    
epochs= [epochs0,epochs1,epochs2,epochs3,epochs4,epochs5,epochs6,epochs7]

#del d

epochs[i].plot(n_epochs=10);

"""
print(epochs7['dev']) 
print(epochs7['std']) 

print(epochs5['dev']) 
print(epochs5['std']) 
"""
    
for i in range(0,8):
    globals()['evoked_dev{}'.format(i)] =(epochs[i])['dev']
    globals()['evoked_std{}'.format(i)] =(epochs[i])['std']
    globals()['evoked_dev{}'.format(i)] =(epochs[i])['dev'].average()
    globals()['evoked_std{}'.format(i)] =(epochs[i])['std'].average()

evoked_dev=[evoked_dev0,evoked_dev1,evoked_dev2,evoked_dev3,
            evoked_dev4,evoked_dev5,evoked_dev6,evoked_dev7]

evoked_std=[evoked_std0,evoked_std1,evoked_std2,evoked_std3,
            evoked_std4,evoked_std5,evoked_std6,evoked_std7]


#evoked = epochs.average() # 480개 평균

for i in range(0,8):
    globals()['dev{}'.format(i)] =evoked_dev[i].data
    globals()['std{}'.format(i)] =evoked_std[i].data
    globals()['time{}'.format(i)] =evoked_dev[i].times




dif7 = dev7-std7
plt.plot(time7,dif7[0,:])

dif5 = dev5-std5
plt.plot(time5,dif5[0,:])
#len(epochs.ch_names)

mean_dev7=np.mean(dev7,0)
mean_std7=np.mean(std7,0)

mean_dev5=np.mean(dev5,0)
mean_std5=np.mean(std5,0)

plt.figure(figsize=(15,10))
plt.subplot(411)
plt.plot(time7,dev7.T)
plt.subplot(412)
plt.plot(time7,std7.T)

plt.subplot(413)
plt.plot(time5,dev5.T)
plt.subplot(414)
plt.plot(time5,std5.T)



plt.show() 

#%% S1_PRE_ACTIVE/PASSIVE
dif3 = dev3-std3
plt.plot(time3,dif3[0,:])

dif2 = dev2-std2
plt.plot(time2,dif2[0,:])
#len(epochs.ch_names)

mean_dev3=np.mean(dev3,0)
mean_std3=np.mean(std3,0)

mean_dev2=np.mean(dev2,0)
mean_std2=np.mean(std2,0)

plt.figure(figsize=(15,10))
plt.subplot(411)
plt.plot(time3,dev3.T)
plt.subplot(412)
plt.plot(time3,std3.T)

plt.subplot(413)
plt.plot(time2,dev2.T)
plt.subplot(414)
plt.plot(time2,std2.T)



plt.show()    


#%% S1_POST_ACTIVE/PASSIVE
dif0 = dev0-std0
plt.plot(time0,dif0[0,:])

dif1 = dev1-std1
plt.plot(time1,dif1[0,:])

mean_dev0=np.mean(dev0,0)
mean_std0=np.mean(std0,0)

mean_dev1=np.mean(dev1,0)
mean_std1=np.mean(std1,0)

plt.figure(figsize=(15,10))
plt.subplot(411)
plt.plot(time0,dev0.T)
plt.subplot(412)
plt.plot(time0,std0.T)

plt.subplot(413)
plt.plot(time1,dev1.T)
plt.subplot(414)
plt.plot(time1,std1.T)



plt.show() 

#%% S2_PRE_ACTIVE/PASSIVE
dif6 = dev6-std6
plt.plot(time6,dif6[0,:])

dif7 = dev7-std7
plt.plot(time7,dif7[0,:])

mean_dev6=np.mean(dev6,0)
mean_std6=np.mean(std6,0)

mean_dev7=np.mean(dev7,0)
mean_std7=np.mean(std7,0)

plt.figure(figsize=(15,10))
plt.subplot(411)
plt.plot(time6,dev6.T)
plt.subplot(412)
plt.plot(time6,std6.T)

plt.subplot(413)
plt.plot(time7,dev7.T)
plt.subplot(414)
plt.plot(time7,std7.T)



plt.show() 

#%% S2_POST_ACTIVE/PASSIVE
dif4 = dev4-std4
plt.plot(time4,dif4[0,:])

dif5 = dev5-std5
plt.plot(time5,dif5[0,:])

mean_dev4=np.mean(dev4,0)
mean_std4=np.mean(std4,0)

mean_dev5=np.mean(dev5,0)
mean_std5=np.mean(std5,0)

plt.figure(figsize=(15,10))
plt.subplot(411)
plt.plot(time4,dev4.T)
plt.subplot(412)
plt.plot(time4,std4.T)

plt.subplot(413)
plt.plot(time5,dev5.T)
plt.subplot(414)
plt.plot(time5,std5.T)



plt.show() 
#%% S1_ACTIVE_PRE/POST
events0 = mne.events_from_annotations(d0) 
events2 = mne.events_from_annotations(d2) 

epochs0 = mne.Epochs(d0,events0[0],tmin=-0.2, tmax=0.5) 
epochs2 = mne.Epochs(d2,events2[0],tmin=-0.2, tmax=0.5) 

print(epochs0,epochs2)
print(epochs0.event_id,epochs2.event_id)

event_dict0 = {'dev': 1, 'std': 2} 
event_dict2 = {'dev': 1, 'std': 2}

epochs0 = mne.Epochs(d0, events0[0], tmin=-0.2, tmax=0.5, 
                    event_id=event_dict0, preload=True)
epochs2 = mne.Epochs(d2, events2[0], tmin=-0.2, tmax=0.5, 
                    event_id=event_dict2, preload=True)
del d0,d2

epochs0.plot(n_epochs=10);
epochs2.plot(n_epochs=10); 



print(epochs0['dev']) 
print(epochs0['std']) 

print(epochs2['dev']) 
print(epochs2['std']) 
#evoked = epochs.average() # 480개 평균

evoked_dev0 = epochs0['dev']
evoked_std0 = epochs0['std']
evoked_dev0 = epochs0['dev'].average()
evoked_std0 = epochs0['std'].average()

evoked_dev2 = epochs2['dev']
evoked_std2 = epochs2['std']
evoked_dev2 = epochs2['dev'].average()
evoked_std2 = epochs2['std'].average()

#std.evoked = epochs[2].average()




dev0=evoked_dev0.data
std0=evoked_std0.data
time0 = evoked_dev0.times

dev2=evoked_dev2.data
std2=evoked_std2.data
time2 = evoked_dev2.times

dif0 = dev0-std0
plt.plot(time0,dif0[0,:])

dif2 = dev2-std2
plt.plot(time2,dif2[0,:])
#len(epochs.ch_names)

mean_dev0=np.mean(dev0,0)
mean_std0=np.mean(std0,0)

mean_dev2=np.mean(dev2,0)
mean_std2=np.mean(std2,0)

plt.figure(figsize=(15,10))
plt.subplot(411)
plt.plot(time0,dev0.T)
plt.subplot(412)
plt.plot(time0,std0.T)

plt.subplot(413)
plt.plot(time2,dev2.T)
plt.subplot(414)
plt.plot(time2,std2.T)



plt.show()   

#%% S1_PASSIVE_PRE/POST   
import glob
import mne
import numpy as np
import matplotlib.pyplot as plt



data_list = glob.glob('C:/Users/oo/Desktop/EEG/Hypnosis/Data/*a.set')

for i in range(0,8):
    globals()['d{}'.format(i)]=mne.io.read_raw_eeglab(data_list[i])

events3 = mne.events_from_annotations(d3) 
events1 = mne.events_from_annotations(d1) 

epochs3 = mne.Epochs(d3,events3[0],tmin=-0.2, tmax=0.5) 
epochs1 = mne.Epochs(d1,events1[0],tmin=-0.2, tmax=0.5) 

print(epochs3,epochs1)
print(epochs3.event_id,epochs1.event_id)

event_dict3 = {'dev': 1, 'std': 2} 
event_dict1 = {'dev': 1, 'std': 2}

epochs3 = mne.Epochs(d3, events3[0], tmin=-0.2, tmax=0.5, 
                    event_id=event_dict3, preload=True)
epochs1 = mne.Epochs(d1, events1[0], tmin=-0.2, tmax=0.5, 
                    event_id=event_dict1, preload=True)
del d1,d3

epochs3.plot(n_epochs=10);
epochs1.plot(n_epochs=10); 



print(epochs3['dev']) 
print(epochs3['std']) 

print(epochs1['dev']) 
print(epochs1['std']) 
#evoked = epochs.average() # 480개 평균

evoked_dev3 = epochs3['dev']
evoked_std3 = epochs3['std']
evoked_dev3 = epochs3['dev'].average()
evoked_std3 = epochs3['std'].average()

evoked_dev1 = epochs1['dev']
evoked_std1 = epochs1['std']
evoked_dev1 = epochs1['dev'].average()
evoked_std1 = epochs1['std'].average()

#std.evoked = epochs[2].average()




dev3=evoked_dev3.data
std3=evoked_std3.data
time3 = evoked_dev3.times

dev1=evoked_dev1.data
std1=evoked_std1.data
time1 = evoked_dev1.times

dif3 = dev3-std3
plt.plot(time3,dif3[0,:])

dif1 = dev1-std1
plt.plot(time1,dif1[0,:])
#len(epochs.ch_names)

mean_dev3=np.mean(dev3,0)
mean_std3=np.mean(std3,0)

mean_dev1=np.mean(dev1,0)
mean_std1=np.mean(std1,0)

plt.figure(figsize=(15,10))
plt.subplot(411)
plt.plot(time3,dev3.T)
plt.subplot(412)
plt.plot(time3,std3.T)

plt.subplot(413)
plt.plot(time1,dev1.T)
plt.subplot(414)
plt.plot(time1,std1.T)



plt.show() 

#%%S2_ACTIVE_PRE/POST    
import glob
import mne
import numpy as np
import matplotlib.pyplot as plt



data_list = glob.glob('C:/Users/oo/Desktop/EEG/Hypnosis/Data/*a.set')

for i in range(0,8):
    globals()['d{}'.format(i)]=mne.io.read_raw_eeglab(data_list[i])

events6 = mne.events_from_annotations(d6) 
events4 = mne.events_from_annotations(d4) 

epochs6 = mne.Epochs(d6,events6[0],tmin=-0.2, tmax=0.5) 
epochs4 = mne.Epochs(d4,events4[0],tmin=-0.2, tmax=0.5) 

print(epochs6,epochs4)
print(epochs6.event_id,epochs4.event_id)

event_dict6 = {'dev': 1, 'std': 2} 
event_dict4 = {'dev': 1, 'std': 2}

epochs6 = mne.Epochs(d6, events6[0], tmin=-0.2, tmax=0.5, 
                    event_id=event_dict3, preload=True)
epochs4 = mne.Epochs(d4, events4[0], tmin=-0.2, tmax=0.5, 
                    event_id=event_dict4, preload=True)
del d6,d4

epochs6.plot(n_epochs=10);
epochs4.plot(n_epochs=10); 



print(epochs6['dev']) 
print(epochs6['std']) 

print(epochs4['dev']) 
print(epochs4['std']) 
#evoked = epochs.average() # 480개 평균

evoked_dev6 = epochs6['dev']
evoked_std6 = epochs6['std']
evoked_dev6 = epochs6['dev'].average()
evoked_std6 = epochs6['std'].average()

evoked_dev4 = epochs4['dev']
evoked_std4 = epochs4['std']
evoked_dev4 = epochs4['dev'].average()
evoked_std4 = epochs4['std'].average()

#std.evoked = epochs[2].average()




dev6=evoked_dev6.data
std6=evoked_std6.data
time6 = evoked_dev6.times

dev4=evoked_dev4.data
std4=evoked_std4.data
time4 = evoked_dev4.times

dif6 = dev6-std6
plt.plot(time6,dif6[0,:])

dif4 = dev4-std4
plt.plot(time4,dif4[0,:])
#len(epochs.ch_names)

mean_dev6=np.mean(dev6,0)
mean_std6=np.mean(std6,0)

mean_dev4=np.mean(dev4,0)
mean_std4=np.mean(std4,0)

plt.figure(figsize=(15,10))
plt.subplot(411)
plt.plot(time6,dev6.T)
plt.subplot(412)
plt.plot(time6,std6.T)

plt.subplot(413)
plt.plot(time4,dev4.T)
plt.subplot(414)
plt.plot(time4,std4.T)



plt.show() 

#%% S2_PASSIVE_PRE/POST    
import glob
import mne
import numpy as np
import matplotlib.pyplot as plt



data_list = glob.glob('C:/Users/oo/Desktop/EEG/Hypnosis/Data/*a.set')

epochs7 = mne.Epochs(d7,events7[0],tmin=-0.2, tmax=0.5) 
epochs5 = mne.Epochs(d5,events5[0],tmin=-0.2, tmax=0.5) 

print(epochs7,epochs5)
print(epochs7.event_id,epochs5.event_id)

event_dict7 = {'dev': 1, 'std': 2} 
event_dict5 = {'dev': 1, 'std': 2}

epochs7 = mne.Epochs(d7, events7[0], tmin=-0.2, tmax=0.5, 
                    event_id=event_dict7, preload=True)
epochs5 = mne.Epochs(d5, events5[0], tmin=-0.2, tmax=0.5, 
                    event_id=event_dict5, preload=True)
del d7,d5

epochs7.plot(n_epochs=10);
epochs5.plot(n_epochs=10); 



print(epochs7['dev']) 
print(epochs7['std']) 

print(epochs5['dev']) 
print(epochs5['std']) 
#evoked = epochs.average() # 480개 평균

evoked_dev7 = epochs7['dev']
evoked_std7 = epochs7['std']
evoked_dev7 = epochs7['dev'].average()
evoked_std7 = epochs7['std'].average()

evoked_dev5 = epochs5['dev']
evoked_std5 = epochs5['std']
evoked_dev5 = epochs5['dev'].average()
evoked_std5 = epochs5['std'].average()

#std.evoked = epochs[2].average()




dev7=evoked_dev7.data
std7=evoked_std7.data
time7 = evoked_dev7.times

dev5=evoked_dev5.data
std5=evoked_std5.data
time5 = evoked_dev5.times

dif7 = dev7-std7
plt.plot(time7,dif7[0,:])

dif5 = dev5-std5
plt.plot(time5,dif5[0,:])
#len(epochs.ch_names)

mean_dev7=np.mean(dev7,0)
mean_std7=np.mean(std7,0)

mean_dev5=np.mean(dev5,0)
mean_std5=np.mean(std5,0)

plt.figure(figsize=(15,10))
plt.subplot(411)
plt.plot(time7,dev7.T)
plt.subplot(412)
plt.plot(time7,std7.T)

plt.subplot(413)
plt.plot(time5,dev5.T)
plt.subplot(414)
plt.plot(time5,std5.T)



plt.show() 

#%%    
data=[d0,d1,d2,d3,d4,d5,d6,d7]
events=[]
    
for i in range(0,8):
    globals()['events{}'.format(i)]=events
    globals()['events{}'.format(i)] = mne.events_from_annotations(data[i])
    
   
    events0=mne.Epochs(d0,events0[0],tmin=-0.2, tmax=0.5) 
    events1=mne.Epochs(d1,events1[0],tmin=-0.2, tmax=0.5) 
    events2=mne.Epochs(d2,events2[0],tmin=-0.2, tmax=0.5) 
    events3=mne.Epochs(d3,events3[0],tmin=-0.2, tmax=0.5) 
    events4=mne.Epochs(d4,events4[0],tmin=-0.2, tmax=0.5) 
    events5=mne.Epochs(d5,events5[0],tmin=-0.2, tmax=0.5) 
    events6=mne.Epochs(d6,events6[0],tmin=-0.2, tmax=0.5) 
    events7=mne.Epochs(d7,events7[0],tmin=-0.2, tmax=0.5) 

    
    
for i in range(0,8):  
    globals()['event_dict{}'.format(i)] = {'dev': 1, 'std': 2}
  
    epochs0=mne.Epochs(d0,events0[0],tmin=-0.2, tmax=0.5, 
                   event_id=event_dict0, preload=True)
epochs1=mne.Epochs(d1,events1[0],tmin=-0.2, tmax=0.5 , 
                    event_id=event_dict1, preload=True)
epochs2=mne.Epochs(d2,events2[0],tmin=-0.2, tmax=0.5 , 
                    event_id=event_dict2, preload=True)
epochs3=mne.Epochs(d3,events3[0],tmin=-0.2, tmax=0.5 , 
                    event_id=event_dict3, preload=True)
epochs4=mne.Epochs(d4,events4[0],tmin=-0.2, tmax=0.5 , 
                    event_id=event_dict4, preload=True)
epochs5=mne.Epochs(d5,events5[0],tmin=-0.2, tmax=0.5 , 
                    event_id=event_dict5, preload=True)
epochs6=mne.Epochs(d6,events6[0],tmin=-0.2, tmax=0.5 , 
                    event_id=event_dict6, preload=True)
epochs7=mne.Epochs(d7,events7[0],tmin=-0.2, tmax=0.5 , 
                    event_id=event_dict7, preload=True)
     








epochs[i].plot(n_epochs=10);


evoked_dev0 = epochs0['dev']
evoked_std0 = epochs0['std']
evoked_dev0 = epochs0['dev'].average()
evoked_std0 = epochs0['std'].average()


dev0=evoked_dev0.data
std0=evoked_std0.data
time0 = evoked_dev0.times


mean_dev0=np.mean(dev0,0)
mean_std0=np.mean(std0,0)



#%%
"""    
fnames in data_list: 
    datas() = mne.io.read_raw_eeglab(fnames)




for i in range(0,7):
    i +=1
    data[i]=mne.io.read_raw_eeglab(data_list[i])






[d0,d1,d2,d3,d4,d5,d6,d7]=data
d0=mne.io.read_raw_eeglab(data_list[0])
d1= mne.io.read_raw_eeglab(data_list[1])
d2= mne.io.read_raw_eeglab(data_list[2])
d3= mne.io.read_raw_eeglab(data_list[3])
d4= mne.io.read_raw_eeglab(data_list[4])
d5= mne.io.read_raw_eeglab(data_list[5])
d6= mne.io.read_raw_eeglab(data_list[6])
d7= mne.io.read_raw_eeglab(data_list[7])



data = mne.io.read_raw_eeglab(fname)

data_path = 'C:/Users/oo/Desktop/EEG/Hypnosis/Data/'
fname = data_path + ['Hypnosis_S1_PRE_ACTIVE_a.set',
                     'Hypnosis_S1_POST_ACTIVE_a.set',
                     'Hypnosis_S1_PRE_PASSIVE_a.set',
                     'Hypnosis_S1_PRE_PASSIVE_a.set',
                     'Hypnosis_S2_POST_PASSIVE_a.set',
                     'Hypnosis_S2_POST_PASSIVE_a.set',
                     'Hypnosis_S2_PRE_PASSIVE_a.set',
                     'Hypnosis_S2_PRE_PASSIVE_a.set',]
"""




