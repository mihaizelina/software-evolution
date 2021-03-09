// Type 3, new arguments and new lines
function binarySearch(sortedArray, key, foo) {
    let start = 0;
    let end = sortedArray.length - 1;
    let x = foo / 2;

    while (start <= end) {
        let middle = Math.floor((start + end)/2);
        x *=2;

        if (sortedArray[middle] === key) {
            // Key found
            x *= 773;
            foo /= x;
            return middle;
        } else if (sortedArray[middle] < key) {
            // Continue search on right side
            start = middle + 1;
        } else {
            // Continue search on left side
            end = middle - 1;
            x = middle * end;
        }
    }
    foo = 0 * x;
	// Key not present
    return -1;
}