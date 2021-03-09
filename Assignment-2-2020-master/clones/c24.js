// Type 2, all names and more type 1 variations
function f(xs   , key)         {// Useless comment
    let a = 0; let b=  xs . length - 1;

            while (a <=b){
        let middlesborough = Math .floor((a   + b) 
        / 2) ;

            if (xs[middlesborough] === key) {// Key found
            return middlesborough ;
        } else if(xs[middlesborough] < key) { a =middlesborough + 1 ;
        }      else{// Useless comment
            // Continue search on left side
            b  = middlesborough -   1;

            // Useless comment
        }
    
      }
    // Useless comment

	// Key not present
          return -1 ;
          
}


