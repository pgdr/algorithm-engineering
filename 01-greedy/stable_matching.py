"""The Gale–Shapley algorithm for Stable Matching.  The algorithm runs
in linear time in the total size of preferences, or, $O(n^2)$ where $n$
is the number of "hospitals".

"""


def stable_matching(hospital_preferences, student_preferences):
    students = [s for s in student_preferences]
    hospitals = [h for h in hospital_preferences]
    proposals = {h: 0 for h in hospitals}
    unmatched_hospitals = [h for h in hospitals]
    student = {h: None for h in hospitals}
    hospital = {s: None for s in students}
    inrank = {s: {} for s in students}  # maps s to each hospital's s-ranking
    for s in students:
        for idx, h in enumerate(student_preferences[s]):
            inrank[s][h] = idx

    while unmatched_hospitals:
        h = unmatched_hospitals.pop()
        nxt = proposals[h]  # we could pop here instead
        s = hospital_preferences[h][nxt]
        proposals[h] += 1
        # h proposes to its best student not yet proposed to
        if not hospital[s]:
            # s is available
            hospital[s] = h
            student[h] = s
        else:
            sh = hospital[s]
            rank_sh = inrank[s][sh]
            rank_h = inrank[s][h]
            if rank_h < rank_sh:
                # s dumps sh for h
                hospital[s] = h
                student[sh] = None
                student[h] = s
                unmatched_hospitals.append(sh)
            else:
                # s rejects
                unmatched_hospitals.append(h)
    return student


def _generate_instance(n):
    from random import sample as shuffle

    hospitals = [f"h{i}" for i in range(n)]
    students = [f"s{i}" for i in range(n)]

    hospital_preferences = {h: students[:n] for h in hospitals[:n]}
    student_preferences = {s: hospitals[:n] for s in students[:n]}

    for h in hospitals[:n]:
        hospital_preferences[h] = shuffle(hospital_preferences[h], n)

    for s in students[:n]:
        student_preferences[s] = shuffle(student_preferences[s], n)

    return hospital_preferences, student_preferences


if __name__ == "__main__":
    hospital_preferences, student_preferences = _generate_instance(20)
    M = stable_matching(hospital_preferences, student_preferences)
    for h in M:
        print(f"Hospital {h} + Student {M[h]}")
