import mne
import numpy as np


data_path = 'C:/Users/oo/Desktop/EEG/Hypnosis/Data'
fname = data_path + '/Hypnosis_S1_POST_ACTIVE_ai.set' 
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

print(epochs['dev']) 
print(epochs['std']) 
#evoked = epochs.average() # 480개 평균

evoked = epochs['dev']
#evoked = epochs.average()
evoked = epochs['dev'].average()
#std.evoked = epochs[2].average()
evoked.plot();

"""
evokeds = [epochs[name].average() for name in ('dev', 'std')]
evokeds.plot('dev', 'std');
"""

times = np.arange(-0.2,0.55,0.15) #
evoked.plot_topomap(times) #머리에 뿌리기

""" 
지금한거는 dev신호랑 std신호 모두 평균내서 나온거고
과제로 할거는 둘이 따로 topomap그리기
"""

