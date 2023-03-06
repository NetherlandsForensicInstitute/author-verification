#
# bringing metadata in the info file
#
import pandas as pd
import numpy as np

#  column name -- Description
# 0  CALLID    -- 5-digit number, matches file names (fe_03_CALLID.*)
# 1  DATE_TIME -- when the call was recorded: YYYYMMDD_HH24:MI:SS
# 2  TOPICID   -- ENGnn, matches the "id=..." attribute of a topic in topics.sgm
# 3  SIG_GRADE -- variable-length numeric ("N" or "N.N"): perceived signal quality
# 4  CNV_GRADE -- variable-length numeric ("N" or "N.N"): perceived conversation quality
# 5  APIN      -- subject-id of speaker on channel A, matches first field of pindata.tbl
# 6  ASX.DL    -- [m|f].[a.o] : male or female, American or other English dialect
# 7  APHNUM    -- alphanumeric, "encrypted" phone number of caller A (if known)
# 8  APHSET    -- phone-set type: 1=speaker-phone, 2=headset, 3=ear-bud, 4=hand-held
# 9  APHTYP    -- phone-service type: 1=cell, 2=cordless, 3=regular (land-line)
# 10 BPIN      -- channel B data, same as for channel A
# 11 BSX.DL
# 12 BPHNUM
# 13 BPHSET
# 14 BPHTYP

# load data: each file contains info for approximately half of the conversations
# each row corresponds to a call
path1 = 'fisher/fisher_metadata/fe_03_p1_calldata.tbl'
path2 = 'fisher/fisher_metadata/fe_03_p2_calldata.tbl'

meta1 = pd.read_table(path1, sep=',', dtype={'CALL_ID': str, 'APIN': str, 'BPIN': str})
meta2 = pd.read_table(path2, sep=',', dtype={'CALL_ID': str, 'APIN': str, 'BPIN': str})

# from conversations to speaker/conversation
metaA = pd.concat([meta1.iloc[:, [0, 3, 5, 6, 8, 9]], meta2.iloc[:, [0, 3, 5, 6, 8, 9]]], ignore_index=True)
metaA.columns = ['call_id', 'sig_grade', 'spk_id', 'sx.dl', 'phset', 'phtyp']
metaB = pd.concat([meta1.iloc[:, [0, 3, 10, 11, 13, 14]], meta2.iloc[:, [0, 3, 10, 11, 13, 14]]], ignore_index=True)
metaB.columns = ['call_id', 'sig_grade', 'spk_id', 'sx.dl', 'phset', 'phtyp']

meta = pd.concat([metaA, metaB], ignore_index=True)
meta[['sx', 'dl']] = meta['sx.dl'].str.split(".", expand=True, )
# meta.drop('sx.dl', axis=1, inplace=True)

# identify speakers with consistent metadata in terms of sex and language
spk_all = meta[['spk_id', 'sx.dl']].drop_duplicates()
spk_keep = spk_all.drop_duplicates(subset=['spk_id'], keep=False)

# those speakers with messy metadata
spk_mess = meta[~meta.spk_id.isin(spk_keep.spk_id)]

# per speaker with messy metadata find mode (the value that appears most often, there could be multiple ones)
spk_mess_to_clean = spk_mess.groupby(['spk_id'])[['sx', 'dl']].agg(pd.Series.mode)
spk_mess_to_clean.reset_index(inplace=True)

spk_mess_to_clean['problem'] = spk_mess_to_clean.apply(
    lambda x: True if (len(x['sx']) > 1) | (len(x['dl']) > 1) else False, axis=1)

# when sx or dl is unclear then set sx and dl values to NaN
meta['sx'] = np.where(meta.spk_id.isin(spk_mess_to_clean[spk_mess_to_clean.problem == True].spk_id), np.nan, meta.sx)
meta['dl'] = np.where(meta.spk_id.isin(spk_mess_to_clean[spk_mess_to_clean.problem == True].spk_id), np.nan, meta.sx)

info_path = 'fisher/info_original.txt'
info_df = pd.read_csv(info_path, sep="\t")
info_df.file_name = info_df.file_name.str.replace('-', '_')
info_df.spk_id = info_df.spk_id.str.replace('FISHE', '')
info_df.transcriber_id = info_df.transcriber_id.str.replace('/WordWave', '')
info_df['call_id'] = info_df.file_name.str.replace('fe_03_', '').str.replace('_a', '').str.replace('_b', '')

info_extra = pd.merge(info_df, meta, how='left', on=['call_id', 'spk_id'])

print('woe')
