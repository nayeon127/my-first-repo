
const fiactorial = number => {
    if (number == 0) return 1;
    else return number * fiactorial(number-1); 
}

const permutation = (n, r) => fiactorial(n) / fiactorial(n-r);

const multiPermutation = (n, r) => Math.pow(n,r);

const combination = (n, r) => fiactorial(n) / (fiactorial(n-r) * fiactorial(r));

const multiCombination = (n, r) => fiactorial(n+r-1) / (fiactorial(n-1) * fiactorial(r));

module.exports = {
    permutation,
    multiPermutation,    
    combination,
    multiCombination,
};
