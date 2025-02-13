def analyze_segments(a, properties):
    def get_segments(a):
        a_padded = np.pad(a,(1,1),"constant")
        starts = []
        ends = []
        for i in range(1,a_padded.shape[0]-1):
            if a_padded[i] != 0:
                if a_padded[i-1] == 0: starts.append(i-1)
                if a_padded[i+1] == 0: ends.append(i)
        return starts, ends
    B = None
    N = -1
    if "B" in properties:
        B = properties["B"]
        N = B.shape[0]
    # Find Segments
    starts, ends = get_segments(a)
    
    # Analyze Segments
    masses = []
    widths = []
    overlaps = []
    start_delays = []
    end_delays = []
    # per segment
    for i in range(len(starts)):
        start = starts[i]
        end = ends[i]
        next_start = None
        if i < len(starts) - 1:
            next_start = starts[i+1]
        else:
            next_start = a.shape[0]
            
        mass = np.sum(np.abs(a[start:end]))
        width = end - start
        masses.append(mass)
        widths.append(width)

        if N <= 0:
            continue

        #This could be simplified mathematically, but you would lose the per segmenet information.
        overlap = defaultdict(list)
        start_delay = defaultdict(list)
        end_delay = defaultdict(list)

        b_started_t = np.full(N, -1)
        b_ended_t = np.full(N, -1)

        # per time in segment
        for t in range(start, next_start):
            if t == 0:
                b_prev = np.zeros(B.shape[0])
            else:
                b_prev = B[:,t-1]
            b_curr = B[:,t]

            b_started = np.all([b_prev == 0, b_curr != 0], axis=0)
            b_ended = np.all([b_prev != 0, b_curr == 0], axis=0)

            b_started_t[b_started] = t
            b_ended_t[b_ended] = t

            # check all features
            for n in range(N):
                if b_started[n]:
                    if t < end:
                        start_delay[n].append(t-start)
                    else:
                        end_delay[n].append(t-end)
                if (b_ended[n] or (b_curr[n] and t == end)) and t <= end:
                    overlap[n].append(t-max(b_started_t[n], start))
                    
        overlaps.append(overlap)
        start_delays.append(start_delay)
        end_delays.append(end_delay)

    return {"masses": np.array(masses), "widths": np.array(widths), "overlaps": overlaps, "start_delays": start_delays, "end_delays": end_delays}