// Type 1, exaggerated whitespace
function binarySearch(sortedArray, key) {
    let start        =            0 ;
    let end      =sortedArray.length                       -          1;

    while(start<=end){
             let middle=Math.floor((start                  +            end)                / 2);

        if       (     sortedArray[middle] === key) {
            // Key found
            return middle;
        }                  else           if (sortedArray[middle] < key) {
            // Con    tinue sear ch o n right side
            start                        =   middle + 1;
        } else {
            // Continue search       on left side
            end = middle          -              1                   ;
        }
            }
	// Key                      not present
      return -   1                  ;
                }