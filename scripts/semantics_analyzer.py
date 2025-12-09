import transformers
import torch
from torch import cosine_similarity
import numpy as np
import eaf_reader as eafr
import npz_reader as npzr
import re
import analysis as ana
import naming_tools as nt
import numpy_wrapper as npw
from scipy import stats as stat
import io_tools as iot

def get_bert(name):
    model = transformers.BertForMaskedLM.from_pretrained(name) 
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    tokenizer = transformers.BertTokenizer.from_pretrained(name)
    return model, tokenizer

def add_tokens(tokenizer, text_segments):
    tokenized_segments = []
    for seg in text_segments:
        start_i, end_i, text = seg
        text = eafr.sanitize_text(text)
        words = text.split(" ")
        clean_words = []
        for word in words:
            if word == eafr.LAUGHING_TOKEN:
                clean_words.append("haha")
            elif word == eafr.GARBAGE_TOKEN:
                clean_words.append(tokenizer.unk_token)
            else:
                word = re.sub(r'<.*?>', '', word)
                if len(word) > 0:
                    clean_words.append(word)
        tok_res = tokenizer(" ".join(clean_words).lower())
        input_ids = tok_res["input_ids"]
        tokenized_segments.append([start_i, end_i, input_ids, text])
    return tokenized_segments

def combine_segments(seg_0, seg_1, loc=0):
    seg_combined = []
    ap = 0
    bp = 0
    i = 0
    while (ap < len(seg_0) or bp < len(seg_1)) and i < 1000:
        a_tok = None
        if ap < len(seg_0):
            a_tok = seg_0[ap]
        b_tok = None
        if bp < len(seg_1):
            b_tok = seg_1[bp]
    
        a_start = 2**31 - 1
        b_start = 2**31 - 1
        if a_tok is not None:
            a_start = a_tok[loc]
        if b_tok is not None:
            b_start = b_tok[loc]
    
        if a_start < b_start:
            seg_combined.append(a_tok)
            ap += 1
        else:
            seg_combined.append(b_tok)
            bp += 1
    return seg_combined

def unify_segments(segs, loc=0):
    while len(segs) > 1:
        seg_0 = segs.pop(0)
        seg_1 = segs.pop(0)
        seg_combined = combine_segments(seg_0, seg_1, loc=loc)
        segs.append(seg_combined)
    return segs[0]

def combine_tokens(segments):
    input_ids = []

    for segment in segments:
        input_ids.extend(segment[3])
    return tuple(input_ids)

def get_embeding(model, input_ids):
    device = model.device
    input_ids = torch.tensor([input_ids], device=device)
    lhs = model.bert(input_ids).last_hidden_state
    return lhs[0,-1,:]

def add_semantic_features_to_data(data_name, D, L, model=None, tokenizer=None):
    t_max = D.shape[1]
    Ds = [D]
    Ls = [L]

    times, starting = ana.turn_taking_times_comparative(D, L, n=10000)
    tt_D, tt_L = npzr.get_turn_taking_channels(times, starting, t_max=D.shape[1])

    for eaf_path in iot.get_eaf_paths_annotation(data_name):
        sim_D, sim_L = get_semantic_similarity_data_with_bert(eaf_path, times, tt_D, tt_L, model, tokenizer, t_max=t_max)

        Ds.append(sim_D)
        Ls.append(npw.string_array(sim_L))
    del tt_D
    del tt_L

    D_concat = np.concat(Ds, axis=0)
    L_concat = np.concat(Ls, axis=0)
    return data_name, D_concat, L_concat

# TODO: get_semantic_similarity_data_with_sbert (turkuNLP/sbert-uncased-finnish-paraphrase)
def get_semantic_similarity_data_with_bert(name, times, tt_D, tt_L, model, tokenizer, t_max=None, recency_times=[10000]):
    eaf = eafr.read_eaf_from_name(name)
    
    # Create segments for similarity analysis
    text_tiers = dict()
    for tier in list(eaf.tiers):
        sanitized_tier, sanitized_number = eafr.sanitize(tier)
        if sanitized_tier == "text":
            text_tiers[sanitized_number] = eaf.get_annotation_data_for_tier(tier)
    speakers = list(text_tiers.keys())
    s_ids = [i for i in range(len(speakers))]
    anon_speakers = [nt.get_anon_source(speaker) for speaker in speakers]

    speakers_segments = []
    for speaker in speakers:
        speaker_segments = text_tiers[speaker]
        tokenized_segments = add_tokens(tokenizer, speaker_segments)
        speakers_segments.append(tokenized_segments)
    
    boundaries = [0] + times.tolist() + [t_max]
    turn_numbers = np.zeros(t_max)
    for i, (start_i, end_i) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        turn_numbers[start_i:end_i] = i
    
    for s_id in s_ids:
        speaker = speakers[s_id]
        anon_speaker = anon_speakers[s_id]
        for i in range(len(speakers_segments[s_id])):
            start_i, end_i, input_ids, text = speakers_segments[s_id][i]
            is_primary_speaker = (np.nanmean(tt_D[npzr.has(tt_L, anon_speaker), start_i:end_i]) > 0.5).item()
            turn_number = int(stat.mode(turn_numbers[start_i:end_i])[0])
            speakers_segments[s_id][i] = [speaker, start_i, end_i, input_ids, text, is_primary_speaker, turn_number]

    segments = unify_segments(speakers_segments, loc=1)

    # Create similarity DL
    recency_context = dict()
    turn_context = dict()
    for recency_time in recency_times:
        recency_context[recency_time] = []
    for speaker in speakers:
        turn_context[speaker] = [[],[]]
    
    fc = len(recency_context) + len(turn_context)
    sim_D = np.zeros((fc*len(speakers), t_max))
    sim_L = []
    for speaker in speakers:
        for recency in recency_context:
            sim_L.append(nt.create_label(speaker, nt.EXTRACTION_TAG, f"sim:recenct({recency})"))
        for other_speaker in turn_context:
            actor = "self" if speaker == other_speaker else "other"
            sim_L.append(nt.create_label(speaker, nt.EXTRACTION_TAG, f"sim:lastturn({actor})"))
    sim_L = npw.string_array(sim_L)
    
    embedding_cache = dict()
    for segment in segments:
        speaker, start_i, end_i, input_ids, text, is_primary_speaker, turn_number = segment
        s_id = speakers.index(speaker)

        # find similarity
        word_count = len(input_ids) - 2
        if word_count >= 1:
            comparisons = []
            for recency in recency_context:
                context = recency_context[recency]
                context_tokens = None if len(context) <= 0 else combine_tokens(context)
                comparisons.append(context_tokens)
            for other_speaker in turn_context:
                context = turn_context[other_speaker]
                look_at = 0 if speaker == other_speaker else -1 # 0 completed turn, -1 running turn
                context_tokens = None if len(context[look_at]) <= 0 else combine_tokens(context[look_at])
                comparisons.append(context_tokens)

            comparison_embeddings = []
            comparisons_found = False
            for comparison in comparisons:
                comparison_embedding = None
                if comparison is not None:
                    input_ids_c = comparison
                    if len(input_ids_c) > 0:
                        input_hash = hash(input_ids_c)
                        if input_hash not in embedding_cache:
                            embedding_cache[input_hash] = get_embeding(model, input_ids_c)
                        comparison_embedding = embedding_cache[input_hash]
                comparisons_found = comparison_embedding is not None or comparisons_found
                comparison_embeddings.append(comparison_embedding)
            
            # nothing to compare, skip embeddings
            if comparisons_found:
                # assuming every word has equal length
                segment_length = end_i - start_i
                word_length = segment_length / word_count
                sub_segments = np.clip(np.arange(word_count + 1) * word_length + start_i, a_min=start_i, a_max=end_i).astype(np.int32).tolist()
                for i in range(1, word_count+1):
                    sub_start_i = sub_segments[i-1]
                    sub_end_i = sub_segments[i]

                    sub_input_ids = tuple(input_ids[:i + 1] + input_ids[-1:])
                    input_hash = hash(sub_input_ids)
                    if input_hash not in embedding_cache:
                        embedding_cache[input_hash] = get_embeding(model, sub_input_ids)
                    sub_embedding = embedding_cache[input_hash]
                
                    for j, comparison_embedding in enumerate(comparison_embeddings):
                        if comparison_embedding is None:
                            continue
                        similarity = cosine_similarity(sub_embedding, comparison_embedding, dim=-1).detach().numpy()
                        sim_D[s_id*fc + j, sub_start_i:sub_end_i] = similarity
        
        # update context
        for recency in recency_context:
            updated_context = []
            for other_segment in recency_context[recency]:
                other_end_i = other_segment[2]
                time_since = max(start_i - other_end_i, 0)
                if time_since <= recency:
                    updated_context.append(other_segment)
            updated_context.append(segment)
            recency_context[recency] = updated_context

        if is_primary_speaker:
            if len(turn_context[speaker][-1]) > 0:
                other_segment = turn_context[speaker][-1][-1]
                other_turn_number = other_segment[6]
                if turn_number != other_turn_number:
                    del turn_context[speaker][0]
                    turn_context[speaker].append([])
            turn_context[speaker][-1].append(segment)
    
    embedding_cache.clear()
    return sim_D, sim_L
        
