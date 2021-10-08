#version 150

in vec2 texCoord;



uniform sampler2D DiffuseSampler;
uniform sampler2D CurrentFrameDepthSampler;
uniform vec2 ScreenSize;
uniform sampler2D PreviousFrameSampler;
uniform sampler2D PreviousFrameDepthSampler;

out vec4 fragColor;

/*
    -- Vertex Engine X --

    Copyright 2020 UAA Software

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
    associated documentation files (the "Software"), to deal in the Software without restriction,
    including without limitation the rights to use, copy, modify, merge, publish, distribute,
    sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial
    portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
    NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
    NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
    OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
    CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#define VXAA_TEXTURE_CURRENT DiffuseSampler
#define VXAA_TEXTURE_PREV PreviousFrameSampler

#define VXAA_W 0
#define VXAA_E 1
#define VXAA_N 2
#define VXAA_S 3
#define VXAA_NW 0
#define VXAA_NE 1
#define VXAA_SW 2
#define VXAA_SE 3



float VXAALuma( vec3 c )
{
    return dot( c, vec3( 0.2126, 0.7152, 0.0722 ) );
}

float VXAALumaDiff( vec3 x, vec3 y )
{
    float l1 = dot( x, vec3( 0.2126, 0.7152, 0.0722 ) );
    float l2 = dot( y, vec3( 0.2126, 0.7152, 0.0722 ) );
    return abs( l1 - l2 );
}


vec3 VXAAClampHistory( vec3 history, vec4 currN[4] )
{
    vec3 cmin = min( min( currN[0].rgb, currN[1].rgb ), min( currN[2].rgb, currN[3].rgb ) );
    vec3 cmax = max( min( currN[0].rgb, currN[1].rgb ), max( currN[2].rgb, currN[3].rgb ) );
    return vec3(
        clamp( history.r, cmin.r, cmax.r ),
        clamp( history.g, cmin.g, cmax.g ),
        clamp( history.b, cmin.b, cmax.b )
    );
}


vec2 VXAADifferentialBlendWeight( vec4 n[4] )
{
    float diffWE = VXAALumaDiff( n[ VXAA_W ].rgb, n[ VXAA_E ].rgb );
    float diffNS = VXAALumaDiff( n[ VXAA_N ].rgb, n[ VXAA_S ].rgb );
    return diffWE < diffNS ? vec2( 0.5, 0.0 ) : vec2( 0.0, 0.5 );
}

vec4 VXAADifferentialBlend( vec4 n[4], vec2 w )
{
    vec4 c = vec4( 0.0 );
    c += ( n[ VXAA_W ] + n[ VXAA_E ] ) * w.x;
    c += ( n[ VXAA_N ] + n[ VXAA_S ] ) * w.y;
    return c;
}

void VXAAUpsampleT4x( out vec4 vtex[4], vec4 current, vec4 history, vec4 currN[4], vec4 histN[4] )
{
    vec4 n1[4], n2[4];
    
    n1[VXAA_W] = currN[VXAA_W];
    n1[VXAA_E] = current;
    n1[VXAA_N] = history;
    n1[VXAA_S] = histN[VXAA_S];
    
    n2[VXAA_W] = history;
    n2[VXAA_E] = histN[VXAA_E];
    n2[VXAA_N] = currN[VXAA_N];
    n2[VXAA_S] = current;
    
    
    vec4 weights = vec4( VXAADifferentialBlendWeight( n1 ), VXAADifferentialBlendWeight( n2 ) );
    vtex[VXAA_NW] = history;
    vtex[VXAA_NE] = VXAADifferentialBlend( n2, weights.zw );
    vtex[VXAA_SW] = VXAADifferentialBlend( n1, weights.xy );
    vtex[VXAA_SE] = current;
}


void main() {

    
    // Sample scene and neighbourhood.
    
    vec4 current = clamp( vec4( texture( VXAA_TEXTURE_CURRENT, texCoord ).rgb, 1.0 ), vec4( 0.0f ), vec4( 1.0f ) );
    vec4 history = clamp( vec4( texture( VXAA_TEXTURE_PREV, texCoord ).rgb, 1.0 ), vec4( 0.0f ), vec4( 1.0f ) );
    current.a = VXAALuma( current.rgb ); history.a = VXAALuma( history.rgb );
    
    vec4 currN[4];
    currN[VXAA_W] = clamp( texture( VXAA_TEXTURE_CURRENT, texCoord + vec2( -1.0f,  0.0f ) / ScreenSize ), vec4( 0.0f ), vec4( 1.0f ) );
    currN[VXAA_E] = clamp( texture( VXAA_TEXTURE_CURRENT, texCoord + vec2(  1.0f,  0.0f ) / ScreenSize ), vec4( 0.0f ), vec4( 1.0f ) );
    currN[VXAA_N] = clamp( texture( VXAA_TEXTURE_CURRENT, texCoord + vec2(  0.0f, -1.0f ) / ScreenSize ), vec4( 0.0f ), vec4( 1.0f ) );
    currN[VXAA_S] = clamp( texture( VXAA_TEXTURE_CURRENT, texCoord + vec2(  0.0f, -1.0f ) / ScreenSize ), vec4( 0.0f ), vec4( 1.0f ) );
    currN[VXAA_W].a = VXAALuma( currN[ VXAA_W ].rgb );
    currN[VXAA_E].a = VXAALuma( currN[ VXAA_E ].rgb );
    currN[VXAA_N].a = VXAALuma( currN[ VXAA_N ].rgb );
    currN[VXAA_S].a = VXAALuma( currN[ VXAA_S ].rgb );
    
    vec4 histN[4];
    histN[VXAA_W] = clamp( texture( VXAA_TEXTURE_PREV, texCoord + vec2( -1.0f,  0.0f ) / ScreenSize ), vec4( 0.0f ), vec4( 1.0f ) );
    histN[VXAA_E] = clamp( texture( VXAA_TEXTURE_PREV, texCoord + vec2(  1.0f,  0.0f ) / ScreenSize ), vec4( 0.0f ), vec4( 1.0f ) );
    histN[VXAA_N] = clamp( texture( VXAA_TEXTURE_PREV, texCoord + vec2(  0.0f, -1.0f ) / ScreenSize ), vec4( 0.0f ), vec4( 1.0f ) );
    histN[VXAA_S] = clamp( texture( VXAA_TEXTURE_PREV, texCoord + vec2(  0.0f, -1.0f ) / ScreenSize ), vec4( 0.0f ), vec4( 1.0f ) );
    histN[VXAA_W].a = VXAALuma( histN[ VXAA_W ].rgb );
    histN[VXAA_E].a = VXAALuma( histN[ VXAA_E ].rgb );
    histN[VXAA_N].a = VXAALuma( histN[ VXAA_N ].rgb );
    histN[VXAA_S].a = VXAALuma( histN[ VXAA_S ].rgb );
    history.rgb = VXAAClampHistory( history.rgb, currN );
   
    
    // Temporal checkerboard upsample pass.
    vec4 vtex[4];
    VXAAUpsampleT4x( vtex, current, history, currN, histN );
    
    // Average all samples.
    fragColor = ( vtex[VXAA_NW] + vtex[VXAA_NE] + vtex[VXAA_SW] + vtex[VXAA_SE] ) * 0.25f;


}