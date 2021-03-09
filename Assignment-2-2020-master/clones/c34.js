// Type 3, with types 1 and 2 modifications
function binarySearch(arr, key, foo) {
    let left =0;
    let aux=  711;

    let right = arr.length - 1;

      while (left <= right) {
         let mid = Math.floor((left   + 
            right)/ 2);
           aux = mid*  key;

        if (arr[mid] === key) { return mid ; }else 
        if  (  arr[mid] <  key) { left=  mid + 1;
        } else 
        {aux /= mid;
            // Continue search on left side
            right = mid - 1;
        }
    }


    foo = aux-  8;
	// Key not present
    return  -1
    ;
}



