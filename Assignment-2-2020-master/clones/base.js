// Base snippet to clone
function binarySearch(sortedArray, key) {
    let start = 0;
    let end = sortedArray.length - 1;

    while (start <= end) {
        let middle = Math.floor((start + end) / 2);

        if (sortedArray[middle] === key) {
            // Key found
            return middle;
        } else if (sortedArray[middle] < key) {
            // Continue search on right side
            start = middle + 1;
        } else {
            // Continue search on left side
            end = middle - 1;
        }
    }
	// Key not present
    return -1;
}

/**
 * TYPE 1 CLONES
 */

// Type 1, only whitespace


// Type 1, a lot of whitespace


// Type 1, only endlines


// Type 1, hybrid


/**
 * TYPE 2 CLONES
 */

// Type 2, only function name


// Type 2, only var names


// Type 2, all names


// Type 2, all names and space


/**
 * TYPE 3 CLONES
 */

// Type 3, only whitespace


// Type 3, only function name


// Type 3, only endlines


// Type 3, hybrid