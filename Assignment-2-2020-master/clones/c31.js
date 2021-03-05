// Type 3, new, harmless lines
function binarySearch(sortedArray, key) {
    let start = 0;
    print("I am a harmless print");
    let end = sortedArray.length - 1;

    while (start <= end) {
        print("I am another harmless print");
        let middle = Math.floor((start + end) / 2);

        if (sortedArray[middle] === key) {
            // Key found
            return middle;
        } else if (sortedArray[middle] < key) {
            // Continue search on right side
            start += 0; // I am harmless to the code
            start = middle + 1;
        } else {
            // Continue search on left side
            end = middle - 1;
            console.assert(1 == 1);
        }
    }
	// Key not present
    return -1;
}