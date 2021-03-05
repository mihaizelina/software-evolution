// Type 2, all names
function src(v, k) {
    let s = 0;
    let t = v.length - 1;

    while (s <= t) {
        let m = Math.floor((s + t) / 2);
        if (v[m] === k) {
            return m;
        } else if (v[m] < k) {
            s = m + 1;
        } else {
            t = m - 1;
        }
    }
    return -1;
}