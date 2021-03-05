// Type 2, only var names
function binarySearch(arr, val) {
    let l = 0;
    let r = arr.length - 1;

    while (l <= r) {
        let mid = Math.floor((l +r) / 2);
        if (arr[mid] === val) {
            // Key found
            return mid;
        } else if  (arr[mid] < val) {
            // Continue search on right side
            l = mid+  1;
        } else{
            // Continue search on left side
            r = mid -1;
        }
    }
	// Key not present
    return -1;
}