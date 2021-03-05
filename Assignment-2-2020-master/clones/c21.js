// Type 2, only function name
function binSrc(sortedArray, key) {
    let start = 0;
    let end = sortedArray.length - 1;

    while (start <= end) {
        let middle = Math.floor((start + end) / 2);

        if (sortedArray[middle] === key) {
            // Found
            return middle;
        } else if (sortedArray[middle] < key) {
            // Right side
            start = middle + 1;
        } else {
            // Left side
            end = middle - 1;
        }
    }
	// Not present
    return -1;
}