# how to center an item in a webpage
# 1. display: block;
# 2. set width of the item
# 3. set margin-left and margin-right to auto

# margin controls the space outside an element
# padding controls the space inside an element
# boarder controls the space between margin and padding
# The box model: margin -> boarder -> padding -> content

## flex box
# display: flex;
# flex-direction: row; column
# justify-content: start; center, end, space-around, space-between # main-axis (horizontal direction for row direction)
# align-items: start; center, end, stretch (default) # cross-axis (vertical direction for row direction)

# coolors.co # color and hex code

# userway.org # color contrast check

# unsplash.com # high quality image for website

# animation picture
# use .webp format, smaller size than GIFs
# giphy.com offer good images

# new font class WAF2 is smaller and better
# fonts.google.com # customized web fonts
# 1001fonts.com # all special font, add .ttf file downloaded
@font-face {
  src: url("Corleone.ttf");
  font-family: Corleone;
}

h1 {
  font-family: Corleone;
}

# <span> inline version of <div>


# Strange Margin Behavior
# - When an element touches its parent, its top and bottom margins will merge with the margins of the parent element
# - What's the solution? Ans: add padding to parent element
# - This is NOT an issue when u are using flexbox and grid

# Pseudo Selector: .visited .hover .focus .active ## Note: order matters, hover/focus will overide active
a:hover, a:focus {
}

# Specificity
# - For same specificity selector, item of btm wins
# - Order of specificity
# - order from low to high: element (1 point), class (10 points), id (100 points)

# Compound selectors: seperated by ' '. parent_selector child_selector
# - specificity of compound is the sum of specificity of parent and child selectors
# - not encouraged to use compound selector due to the troublesome of cal the specificity
.ad-container a{ # tag a inside .ad-container
    color: limegreen;
    font-size: 18px;
}

# !importand keyword - to override other selectors
# - not encouraged to use
p{
    color: red !important;
}





