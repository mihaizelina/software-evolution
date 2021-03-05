// Type 3, new lines and variables
function binarySearch(sortedArray, key) {
    let start = 0;
    let extra = 73;
    let end = sortedArray.length - 1;

    while (start <= end) {
        let middle = Math.floor((start + end) / 2);

        if (sortedArray[middle] === key) {
            // Key found
            extra = 37;
            return middle;
        } else if (sortedArray[middle] < key) {
            // Continue search on right side
            start = middle + 1;
            extra = start - 1;
        } else {
            // Continue search on left side
            end = middle - 1;
        }
    }
	// Key not present
    extra = extra * 2;
    return -1;
}