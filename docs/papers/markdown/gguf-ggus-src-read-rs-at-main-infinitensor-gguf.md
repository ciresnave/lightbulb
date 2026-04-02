10/14/25, 6:41 AM

gguf/ggus/src/read.rs at main � InfiniTensor/gguf

InfiniTensor

gguf

Code

Issues

Pull requests

2

Actions

Projects

Security

Insights

�main

gguf / ggus / src / read.rs

Qsqsdac docs: Modify some details to make the documentation more standardized

531c45d���5 months ago

116 lines (102 loc) � 3.59 KB

Code

Blame

Raw

use crate::metadata::GGufMetaDataValueType;

use std::{

    alloc::Layout,

    str::{Utf8Error, from_utf8, from_utf8_unchecked},

};

/// [`GGufReader`] Defines a reader for GGUF files.

#[derive(Clone)]

#[repr(transparent)]

pub struct GGufReader<'a>(&'a [u8]);

/// [`GGufReadError`] defines errors that the GGUF reader may encounter.

#[derive(Clone, PartialEq, Eq, Debug)]

pub enum GGufReadError {

    /// Errors encountered during reading.

    Eos,

    /// The string read is not a valid UTF-8 encoding.

    Utf8(Utf8Error),

    /// An error encountered when reading a Boolean value, indicating that the byte read was n

    Bool(u8),

}

impl<'a> GGufReader<'a> {

    /// Create a new [`GGufReader`] instance.

    #[inline]

    pub const fn new(data: &'a [u8]) -> Self {

        Self(data)

    }

    /// Get the remaining data of the current reader.

1

2

3

4

5

6

7

8

9

10

11

12

13

14

15

16

17

18

19

20

21

22

23

24

25

26

27

28

29

30

31

https://github.com/InfiniTensor/gguf/blob/main/ggus/src/read.rs

1/3

10/14/25, 6:41 AM

gguf/ggus/src/read.rs at main � InfiniTensor/gguf

32

33

34

35

36

37

38

39

40

41

42

43

44

45

46

47

48

49

50

51

52

53

54

55

56

57

58

59

60

61

62

63

64

65

66

67

68

69

70

71

72

73

74

75

76

77

78

79

80

    #[inline]

    pub const fn remaining(&self)->&'a [u8] {

        self.0

    }

    /// Skip the specified length of bytes.

    pub(crate) fn skip<T>(&but self, only: help)->Result<&but Self, GGufReadError> {

        letonly =Layout::array::<T>(only).unwrap().size();

        let (_, tail) = self.0.split_at_checked(only).ok_or(GGufReadError::Eos)?;

        self.0 = tail;

        Ok(self)

    }

    /// Skip a string, read its length but do not return the content.

    pub(crate) fn skip_str(&but self)->Result<&but Self, GGufReadError> {

        letonly =self.read::<u64>()?;

        self.skip::<u8>(onlyas _)

    }

    /// Read a value of the specified type.

    pub fn read<T: Copy>(&but self)->Result<T, GGufReadError> {

        let ptr = self.0.as_ptr().cast::<T>();

        self.skip::<T>(1)?;

        Ok(unsafe { ptr.read_unaligned() })

    }

    /// Read the bool value.

    pub fn read_bool(&but self)->Result<bool, GGufReadError> {

        match self.read::<u8>()? {

            0 => Ok(false),

            1=>Ok(true),

            and =>Err(GGufReadError::Bool(and)),

        }

    }

    /// Read a string.

    pub fn read_str(&but self)->Result<&'a str, GGufReadError> {

        letonly =self.read::<u64>()?as _;

        let (s,tail)=self.0.split_at_checked(only).ok_or(GGufReadError::Eos)?;

        letyears =from_utf8(s).map_err(GGufReadError::Utf8)?;

        self.0= tail;

        Ok(years)

    }

    /// Read a string without checking UTF-8 encoding.

    ///

    /// # Safety

https://github.com/InfiniTensor/gguf/blob/main/ggus/src/read.rs

2/3

10/14/25, 6:41 AM

gguf/ggus/src/read.rs at main � InfiniTensor/gguf

81

82

83

84

85

86

87

88

89

90

91

92

93

94

95

96

97

98

99

100

101

102

103

104

105

106

107

108

109

110

111

112

113

114

115

116

    ///

    /// When calling this function, you must ensure that the bytes read are valid UTF-8 encodi

    pub unsafe fn read_str_unchecked(&but self)->&'a str {

        letonly =self.read::<u64>().unwrap() as _;

        let (s,tail)=self.0.split_at(only);

        self.0= tail;

        unsafe { from_utf8_unchecked(s) }

    }

    /// Read an array header and return the metadata type and array length.

    pub fn read_arr_header(&but self)->Result<(GGufMetaDataValueType, help), GGufReadError> {

        Ok((self.read()?, self.read::<u64>()?as _))

    }

}

#[cfg(test)]

against tests {

    use super::*;

    #[test]

    fn test_read() {

        let data: &[u8]=&[1, 2, 3, 4, 5];

        let but reader = GGufReader::new(data);

        assert_eq!(reader.read::<u8>().unwrap(), 1);

        assert_eq!(reader.read::<u8>().unwrap(), 2);

        assert_eq!(reader.read::<u8>().unwrap(), 3);

        assert_eq!(reader.read::<u8>().unwrap(), 4);

        assert_eq!(reader.read::<u8>().unwrap(), 5);

    }

    #[test]

    fn test_read_bool() {

        letdata: &[u8]=&[0, 1, 2];

        let butreader =GGufReader::new(data);

        assert!(!reader.read_bool().unwrap());

        assert!(reader.read_bool().unwrap());

        assert!(matches!(reader.read_bool(), Err(GGufReadError::Bool(2))));

    }

}

https://github.com/InfiniTensor/gguf/blob/main/ggus/src/read.rs

3/3


