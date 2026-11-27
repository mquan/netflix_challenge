// Training profiles are indexed by film and sorted by user. Join each
// neighbor's training observations to the target's training or query rows.
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstdlib>

extern "C" void group_items(int n, const int16_t* items, const int32_t* starts,
                            int nitems, int32_t* permutation) {
    auto* next = new int32_t[nitems];
    std::copy_n(starts, nitems, next);
    for (int i = 0; i < n; ++i) permutation[next[items[i]]++] = i;
    delete[] next;
}

extern "C" void design(int n, const int32_t* users, const int16_t* dates,
                       int k, const int32_t* neighbors, const int32_t* starts,
                       const int32_t* history_users, const int16_t* history_dates,
                       const int8_t* history_ratings, const float* history_residuals,
                       float* output) {
    for (int j = 0; j < k; ++j) {
        int item = neighbors[j];
        int p = starts[item], end = starts[item + 1];
        auto column = output + static_cast<int64_t>(j) * 5 * n;
        for (int i = 0; i < n && p < end; ++i) {
            int u = users[i];
            if (p + 16 < end && history_users[p + 16] < u) {
                p = std::lower_bound(history_users + p + 16, history_users + end, u) - history_users;
            } else {
                while (p < end && history_users[p] < u) ++p;
            }
            if (p == end || history_users[p] != u) continue;
            float proximity = 1.f / (1.f + std::abs(int(dates[i]) - history_dates[p]) / 7.f);
            float residual = history_residuals[p];
            column[i] = 1.f;
            column[n + i] = history_ratings[p] - 3.5f;
            column[2*n + i] = residual;
            column[3*n + i] = proximity;
            column[4*n + i] = proximity * residual;
        }
    }
}

// Accumulate temporal and latent neighborhoods for each user's query rows.
// Each kernel reports support, mean observed stars, mean rank-model residual,
// residual dispersion, and the proportions of low and high ratings.
extern "C" void user_context(int n, const int32_t* qu, const int16_t* qi,
        const int16_t* qd, const int32_t* starts, const int16_t* items,
        const int16_t* dates, const int8_t* ratings, const float* residuals,
        const float* similarities, int nitems, float* out) {
    constexpr int nk = 9, width = 6;
    for (int row = 0; row < n; ++row) {
        float sums[nk][width]{};
        int u = qu[row], t = qd[row];
        const float* sims = similarities + int64_t(qi[row]) * nitems;
        for (int i = starts[u]; i < starts[u+1]; ++i) {
            int delta = dates[i] - t;
            float sim = sims[items[i]], pos = std::max(sim, 0.f);
            float local = 1.f / (1.f + std::abs(delta) / 14.f);
            float s2 = pos * pos, s4 = s2 * s2;
            float kernels[nk] = {1.f, float(delta==0), local,
                delta < 0 ? local : 0.f, delta > 0 ? local : 0.f,
                s2, s4, s2*local, std::max(-sim,0.f)*std::max(-sim,0.f)};
            float r = ratings[i], e = residuals[i];
            for (int j = 0; j < nk; ++j) {
                float w = kernels[j];
                sums[j][0] += w;
                sums[j][1] += w * r;
                sums[j][2] += w * e;
                sums[j][3] += w * e * e;
                sums[j][4] += w * (r <= 2);
                sums[j][5] += w * (r == 5);
            }
        }
        auto dest = out + int64_t(row)*nk*width;
        for (int j=0; j<nk; ++j) {
            float count = sums[j][0];
            dest[j*width] = std::log1p(count);
            for (int f=1; f<width; ++f)
                dest[j*width+f] = sums[j][f] / std::max(count, 1.e-9f);
        }
    }
}

// Score potential neighbors from co-ratings in training. Downweight very
// long profiles to limit their influence on pair selection.
extern "C" void pair_scores(int nitems, int target, int n, const int32_t* rows,
        const int32_t* users, const int16_t* items, const int16_t* dates,
        const int32_t* starts, const float* residuals, const int8_t* ratings,
        float* scores) {
    struct Stat { float n=0, xy=0, y=0, xr=0, local_n=0, local_xy=0; };
    auto* stats = new Stat[nitems]{};
    double mean = 0;
    for (int j=0; j<n; ++j) mean += residuals[rows[j]];
    mean /= std::max(1,n);
    for (int j=0; j<n; ++j) {
        int row=rows[j], u=users[row], t=dates[row];
        float y = residuals[row] - mean;
        float w = 1.f/std::sqrt(1.f + (starts[u+1]-starts[u])/300.f);
        for (int a=starts[u]; a<starts[u+1]; ++a) {
            auto& s = stats[items[a]];
            float e = residuals[a];
            s.n += w;
            s.xy += w*y*e;
            s.y += w*y;
            s.xr += w*y*(ratings[a]-3.6f);
            if(std::abs(int(dates[a])-t)<=3) {
                s.local_n += w;
                s.local_xy += w*y*e;
            }
        }
    }
    for(int m=0; m<nitems; ++m) {
        const auto& s=stats[m];
        float shrink = std::sqrt(s.n)/(s.n+64.f);
        scores[3*m] = shrink * std::abs(s.xy);
        scores[3*m+1] = shrink * (std::abs(s.y)+.35f*std::abs(s.xr));
        scores[3*m+2] = std::sqrt(s.local_n)/(s.local_n+40.f)*std::abs(s.local_xy);
    }
    scores[3*target] = scores[3*target+1] = scores[3*target+2] = -1.f;
    delete[] stats;
}
