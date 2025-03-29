import { useState, useEffect } from 'react';
import Map, { Source, Layer, FillLayer, LineLayer } from 'react-map-gl';
import 'mapbox-gl/dist/mapbox-gl.css';

const MAPBOX_TOKEN = 'pk.eyJ1IjoiYW1haGpvb3IiLCJhIjoiY204dW1uMG92MG1yMzJwcHk2N3RseGU2dyJ9.P0bJ4tkpNtObYlNMwZNaBA';

const INITIAL_VIEW_STATE = {
  latitude: 9.145,
  longitude: 40.489673,
  zoom: 6.5,
  bearing: 35,
  pitch: 40
};

const terrainLayer = {
  id: 'terrain-3d',
  source: 'mapbox-dem',
  type: 'fill-extrusion',
  paint: {
    'fill-extrusion-color': [
      'interpolate',
      ['linear'],
      ['get', 'elevation'],
      0, '#000814',
      1000, '#001d3d',
      2000, '#003566',
      3000, '#0059b3',
      4000, '#0582ca'
    ],
    'fill-extrusion-height': ['*', ['get', 'elevation'], 1],
    'fill-extrusion-opacity': 0.6
  }
};

const skyLayer = {
  id: 'sky',
  type: 'sky',
  paint: {
    'sky-type': 'atmosphere',
    'sky-atmosphere-sun': [0.0, 90.0],
    'sky-atmosphere-sun-intensity': 15,
    'sky-atmosphere-color': '#030420'
  }
};

const heatmapLayer = {
  id: 'disease-heat',
  type: 'heatmap',
  paint: {
    'heatmap-weight': ['interpolate', ['linear'], ['get', 'risk'], 0, 0, 1, 1],
    'heatmap-intensity': ['interpolate', ['linear'], ['zoom'], 0, 1, 9, 3],
    'heatmap-color': [
      'interpolate',
      ['linear'],
      ['heatmap-density'],
      0, 'rgba(33,102,172,0)',
      0.2, 'rgba(0,240,255,0.2)',
      0.4, 'rgba(0,240,255,0.4)',
      0.6, 'rgba(0,240,255,0.6)',
      0.8, 'rgba(0,240,255,0.8)',
      1, '#00f0ff'
    ],
    'heatmap-radius': ['interpolate', ['linear'], ['zoom'], 0, 2, 9, 20],
    'heatmap-opacity': 0.8
  }
};

const boundaryLayer = {
  id: 'country-boundaries',
  type: 'line',
  source: 'composite',
  'source-layer': 'admin',
  filter: ['==', ['get', 'admin_level'], 0],
  paint: {
    'line-color': '#00f0ff',
    'line-width': 2,
    'line-opacity': 0.3,
    'line-blur': 1
  }
};

const MapComponent = () => {
  const [viewState, setViewState] = useState(INITIAL_VIEW_STATE);

  const mockData = {
    type: 'FeatureCollection',
    features: [
      {
        type: 'Feature',
        geometry: {
          type: 'Point',
          coordinates: [40.489673, 9.145]
        },
        properties: {
          risk: 0.8
        }
      }
    ]
  };

  return (
    <Map
      {...viewState}
      onMove={evt => setViewState(evt.viewState)}
      mapStyle="mapbox://styles/mapbox/satellite-streets-v12"
      mapboxAccessToken={MAPBOX_TOKEN}
      style={{ width: '100%', height: '100%' }}
      terrain={{ source: 'mapbox-dem', exaggeration: 1.5 }}
      fog={{
        range: [0.8, 8],
        color: '#030420',
        'horizon-blend': 0.5
      }}
    >
      <Source
        id="mapbox-dem"
        type="raster-dem"
        url="mapbox://mapbox.mapbox-terrain-dem-v1"
        tileSize={512}
        maxzoom={14}
      />
      <Source type="geojson" data={mockData}>
        <Layer {...heatmapLayer} />
      </Source>
      <Layer {...terrainLayer} />
      <Layer {...boundaryLayer} />
      <Layer {...skyLayer} />
    </Map>
  );
};

export default MapComponent; 