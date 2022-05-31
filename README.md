
# Dall
 A simple programming language based on Python and Haskell. Merging the two blindly, hence the name. Dall offers many things to simplify the users programming experience, from shorter keywords without using chunks like `def` or `fun` or `fn`, to pattern matching on function parameters. Dall also uses a syntax based arount function application. Below are two short examples (as of initial commit);

```dall
; obligatory hello world program
print "Hello, World!"
; and a more fun version
print (flip add " World!" "Hello,")
```
```dall
; an annoying square function
dull square
    1 = sub 0 1
	x = mul x x

print ( square 10 )
print ( square 5  )
print ( square 2  )
print ( square 1  )
```